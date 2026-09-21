#!/usr/bin/env python3
"""基于 train_no_unicore.py 的本地评测: 加载 HPC checkpoint, 对 valid LMDB 推理, 导出预测 CSV
用法: eval_ckpt_no_unicore.py <data_dir> --restore-file <ckpt.pt> --save-dir <out> --encoder-layers N
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import r2_score

ROOT = "/home/sakura/app/unimol_no_unicore"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, os.path.join(ROOT, "unicore_compat"))

import train_no_unicore as T  # noqa: E402
from unicore_compat import checkpoint_utils  # noqa: E402
from unicore_compat.tasks import setup_task  # noqa: E402
import unimol.tasks.unimol_finetune  # noqa: E402
import unimol.models.unimol_test  # noqa: E402
import unimol.losses.reg_loss  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("eval")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="LMDB data dir (train.lmdb + valid.lmdb)")
    parser.add_argument("--restore-file", required=True, help="checkpoint path")
    parser.add_argument("--save-dir", default="./eval_out")
    parser.add_argument("--encoder-layers", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.save_dir = args.save_dir
    os.makedirs(args.save_dir, exist_ok=True)

    # 用训练默认参数构建完整 args (setup_task 需要), 不经过 train parser
    import argparse as _ap
    full = _ap.Namespace()
    # checkpoint 里的训练 args 优先
    ckpt_state = checkpoint_utils.load_checkpoint_to_cpu(args.restore_file)
    ckpt_args = ckpt_state.get("args", {})
    if isinstance(ckpt_args, dict):
        for k, v in ckpt_args.items():
            if not k.startswith("_"):
                setattr(full, k, v)
    else:
        full = ckpt_args
    # 显式覆盖
    full.data = args.data
    full.save_dir = args.save_dir
    full.encoder_layers = args.encoder_layers
    full.batch_size = args.batch_size
    full.num_workers = args.num_workers
    full.device = args.device
    # 注意: mode 必须是 "train" 才能让 forward 返回 logits (mode="infer" 返回 encoder_rep)
    full.mode = "train"
    # 不依赖远程预训练权重 (build_model 会尝试加载 finetune_from_model)
    full.finetune_from_model = None
    full = T.build_args(full)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    task = setup_task(full)
    task.load_dataset("valid")
    model = task.build_model(full).to(device)
    criterion = task.build_criterion(full).to(device)

    # 加载 checkpoint
    if "model" in ckpt_state:
        model.load_state_dict(ckpt_state["model"], strict=False)
    logger.info("Checkpoint loaded: %s (epoch=%s)", Path(args.restore_file).name, ckpt_state.get("epoch", "?"))

    model.eval()
    criterion.eval()
    valid_loader = task.get_batch_iterator(
        task.datasets["valid"], batch_size=full.batch_size,
        shuffle=False, num_workers=full.num_workers)

    all_predicts, all_targets = [], []
    with torch.no_grad():
        for sample in valid_loader:
            for key in sample.get("net_input", {}):
                if isinstance(sample["net_input"][key], torch.Tensor):
                    sample["net_input"][key] = sample["net_input"][key].to(device)
            if "target" in sample:
                for key in sample["target"]:
                    if isinstance(sample["target"][key], torch.Tensor):
                        sample["target"][key] = sample["target"][key].to(device)
            loss, sample_size, logging_output = criterion(model, sample, reduce=True)
            if "predict" in logging_output:
                all_predicts.append(logging_output["predict"].cpu().numpy())
                all_targets.append(logging_output["target"].cpu().numpy())

    y_pred = np.concatenate(all_predicts, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    num_targets = y_pred.shape[1] if y_pred.ndim > 1 else 1
    for j in range(num_targets):
        yp = y_pred[:, j] if num_targets > 1 else y_pred
        yt = y_true[:, j] if num_targets > 1 else y_true
        logger.info("  target_%d  MAE %.4f  R² %.4f", j, float(np.abs(yp - yt).mean()),
                    float(r2_score(yt, yp)))
    agg = float(np.abs(y_pred - y_true).mean())
    logger.info("  AGGREGATE  MAE %.4f", agg)

    # 导出预测 CSV
    out = os.path.join(args.save_dir, "predictions.csv")
    np.savez(out.replace(".csv", ".npz"), y_pred=y_pred, y_true=y_true)
    # 也写 CSV
    import pandas as pd
    cols = {"true_0": y_true[:, 0] if num_targets > 1 else y_true,
            "pred_0": y_pred[:, 0] if num_targets > 1 else y_pred}
    if num_targets > 1:
        cols["true_1"] = y_true[:, 1]
        cols["pred_1"] = y_pred[:, 1]
    pd.DataFrame(cols).to_csv(out, index=False)
    logger.info("Saved: %s", out)


if __name__ == "__main__":
    main()
