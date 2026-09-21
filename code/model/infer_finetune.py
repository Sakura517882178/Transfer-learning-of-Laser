#!/usr/bin/env python3
"""UniMol no-unicore 批量推理 — 读 LMDB (训练同源), 输出 R²/MAE + 预测 CSV。

与 train_no_unicore.py 使用同一数据管线 (unicore_compat task):
  - task.load_dataset('valid') → get_batch_iterator 批量
  - criterion(model, sample) 输出反归一化 predict
用法:
  python infer_no_unicore_finetune.py --work-dir <work_dir> [--device cuda] [--output out.csv]
work_dir 需含: save_finetune/checkpoint_best.pt, {task_name}/train.lmdb+valid.lmdb, pretrain_dict.txt
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_absolute_error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from unicore_compat import checkpoint_utils  # noqa: E402
from unicore_compat.tasks import setup_task  # noqa: E402
import unimol.tasks.unimol_finetune  # noqa: E402
import unimol.models.unimol_test  # noqa: E402
import unimol.losses.reg_loss  # noqa: E402


def build_args(state):
    import argparse as ap
    args = ap.Namespace()
    ckpt_args = state.get("args", None)
    if isinstance(ckpt_args, dict):
        for k, v in ckpt_args.items():
            if not k.startswith("_"):
                setattr(args, k, v)
    elif ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            setattr(args, k, v)
    defaults = {
        "encoder_layers": 15, "encoder_embed_dim": 512, "encoder_ffn_embed_dim": 2048,
        "encoder_attention_heads": 64, "dropout": 0.1, "emb_dropout": 0.1,
        "attention_dropout": 0.1, "activation_dropout": 0.0, "pooler_dropout": 0.0,
        "max_seq_len": 512, "activation_fn": "gelu", "pooler_activation_fn": "tanh",
        "post_ln": False, "masked_token_loss": -1.0, "masked_coord_loss": -1.0,
        "masked_dist_loss": -1.0, "x_norm_loss": -1.0,
        "delta_pair_repr_norm_loss": -1.0, "mode": "train",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--checkpoint", default=None, help="checkpoint 路径 (默认 work_dir/save_finetune/checkpoint_best.pt)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default=None, help="Save predictions CSV")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    wd = Path(args.work_dir)

    ckpt = Path(args.checkpoint) if args.checkpoint else (wd / "save_finetune" / "checkpoint_best.pt")
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        return
    d_path = wd / "pretrain_dict.txt"
    if not d_path.exists():
        print(f"Dict not found: {d_path}")
        return

    state = checkpoint_utils.load_checkpoint_to_cpu(str(ckpt))
    full = build_args(state)
    # 覆盖: 数据/推理参数
    full.data = str(wd)
    full.save_dir = str(wd / "save_finetune")
    full.device = args.device
    full.fp16 = False
    full.num_workers = 0
    full.no_shuffle = True
    full.batch_size = args.batch_size
    full.finetune_from_model = None
    full.mode = "train"  # forward 返回 logits (mode='infer' 返回 encoder_rep)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Loading: {ckpt.name} (L{full.encoder_layers})")

    task = setup_task(full)
    task.load_dataset("valid")
    model = task.build_model(full).to(device)
    model.load_state_dict(state["model"], strict=False)
    model.eval()
    criterion = task.build_criterion(full).to(device).eval()

    loader = task.get_batch_iterator(task.datasets["valid"], batch_size=full.batch_size,
                                     shuffle=False, num_workers=0)
    n_total = len(task.datasets["valid"])
    print(f"valid: {n_total} mol, batch={full.batch_size}")

    all_pred, all_true, all_names = [], [], []
    with torch.no_grad():
        for sample in tqdm(loader, desc="valid"):
            net = sample["net_input"]
            for k, v in net.items():
                if isinstance(v, torch.Tensor):
                    net[k] = v.to(device)
            for k, v in sample.get("target", {}).items():
                if isinstance(v, torch.Tensor):
                    sample["target"][k] = v.to(device)
            _, _, logging_output = criterion(model, sample, reduce=True)
            if "predict" in logging_output:
                all_pred.append(logging_output["predict"].cpu().numpy())
                all_true.append(sample["target"]["finetune_target"].cpu().numpy())
            if "name" in logging_output:
                all_names.extend(logging_output["name"])

    y_pred = np.concatenate(all_pred)
    y_true = np.concatenate(all_true)
    target_names = getattr(full, "target_names", None)
    if target_names is None:
        target_names = [f"target_{i}" for i in range(y_pred.shape[1])]
    print(f"\n=== valid ({len(y_pred)} mol) ===")
    for j, t in enumerate(target_names[:y_pred.shape[1]]):
        mae = mean_absolute_error(y_true[:, j], y_pred[:, j])
        r2 = r2_score(y_true[:, j], y_pred[:, j])
        print(f"  {t:<12} MAE {mae:.4f}  R² {r2:.4f}")
    agg_mae = np.abs(y_pred - y_true).mean()
    print(f"  AGGREGATE     MAE {agg_mae:.4f}")

    if args.output:
        import pandas as pd
        data = {"name": all_names}
        for j, t in enumerate(target_names[:y_pred.shape[1]]):
            data[f"{t}_true"] = y_true[:, j]
            data[f"{t}_pred"] = y_pred[:, j]
        pd.DataFrame(data).to_csv(args.output, index=False)
        print(f"  Saved: {args.output}")


if __name__ == "__main__":
    main()
