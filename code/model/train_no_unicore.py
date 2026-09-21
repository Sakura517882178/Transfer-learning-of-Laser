#!/usr/bin/env python3
"""Native PyTorch training loop for UniMol finetuning without unicore.

Matches the unicore-train parameter set: polynomial_decay LR scheduler,
Adam (betas 0.9,0.99 eps 1e-6), FP16 AMP, early stopping, freeze layers,
per-target R²/MAE, consistent loss.
"""

import argparse
import logging
import os
import sys
import math

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import r2_score

from unicore_compat import checkpoint_utils, distributed_utils
from unicore_compat.tasks import setup_task
import unimol.tasks.unimol_finetune
import unimol.models.unimol_test
import unimol.losses.reg_loss

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser("Train UniMol without unicore")
    parser.add_argument("data", help="path to data directory")
    # ── task / model ──
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--task", default="mol_finetune")
    parser.add_argument("--loss", default="finetune_smooth_mae")
    parser.add_argument("--arch", default="unimol_base")
    parser.add_argument("--dict-name", default="pretrain_dict.txt")
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--classification-head-name", default="classification")
    # ── architecture ──
    parser.add_argument("--encoder-layers", type=int, default=15)
    parser.add_argument("--encoder-embed-dim", type=int, default=512)
    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=2048)
    parser.add_argument("--encoder-attention-heads", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="pooler dropout")
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--activation-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-atoms", type=int, default=510)
    # ── optimizer ──
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--adam-betas", default="(0.9, 0.99)")
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    # ── LR scheduler ──
    parser.add_argument("--lr-scheduler", default="polynomial_decay",
                       choices=["polynomial_decay", "fixed"])
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--lr-power", type=float, default=1.0,
                       help="polynomial decay power")
    # ── training ──
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default="./checkpoints")
    parser.add_argument("--finetune-from-model", default=None)
    parser.add_argument("--no-shuffle", action="store_true")
    parser.add_argument("--conf-size", type=int, default=1)
    # ── early stopping / validation ──
    parser.add_argument("--patience", type=int, default=20,
                       help="early stopping patience (0=disabled)")
    parser.add_argument("--validate-interval", type=int, default=1)
    parser.add_argument("--best-checkpoint-metric", default="valid_agg_mae")
    # ── fp16 ──
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--fp16-init-scale", type=int, default=4)
    parser.add_argument("--fp16-scale-window", type=int, default=256)
    # ── freeze / molecular ──
    parser.add_argument("--freeze-encoder-layers", type=int, default=0,
                       help="freeze first N encoder layers (0=disabled)")
    parser.add_argument("--remove-polar-hydrogen", action="store_true")
    parser.add_argument("--remove-hydrogen", action="store_true")
    parser.add_argument("--only-polar", type=int, default=1)
    parser.add_argument("--consistent-loss", type=float, default=0.0)
    parser.add_argument("--weight-constant", type=float, default=20.0)
    # ── logging ──
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--no-epoch-checkpoints", action="store_true",
                       help="save only best checkpoint")
    return parser


def build_args(args):
    defaults = {
        "encoder_ffn_embed_dim": 2048,
        "encoder_attention_heads": 64,
        "emb_dropout": 0.1,
        "pooler_dropout": 0.0,
        "activation_fn": "gelu",
        "pooler_activation_fn": "tanh",
        "post_ln": False,
        "masked_token_loss": -1.0,
        "masked_coord_loss": -1.0,
        "masked_dist_loss": -1.0,
        "x_norm_loss": -1.0,
        "delta_pair_repr_norm_loss": -1.0,
        "mode": "train",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def parse_betas(s: str):
    s = s.strip("()")
    parts = [float(x.strip()) for x in s.split(",")]
    return tuple(parts)


def _lr_polynomial_decay(base_lr, step, total_steps, warmup_steps, power=1.0):
    if step < warmup_steps:
        return base_lr * float(step) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * max(0.0, (1.0 - progress)) ** power


def train_epoch(epoch, task, model, criterion, optimizer, scaler, train_loader,
                device, total_steps, warmup_steps, args):
    model.train()
    criterion.train()
    task.args.epoch = epoch
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)

    total_loss = 0.0
    total_samples = 0
    lr = optimizer.param_groups[0]["lr"]
    pbar = tqdm(train_loader, desc=f"Epoch {epoch:3d} (lr={lr:.2e})", unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} "
                           "[{elapsed}<{remaining}, {rate_fmt}] {postfix}",
                file=sys.stdout, dynamic_ncols=True)
    for i, sample in enumerate(pbar):
        for key in sample.get("net_input", {}):
            if isinstance(sample["net_input"][key], torch.Tensor):
                sample["net_input"][key] = sample["net_input"][key].to(device)
        if "target" in sample:
            for key in sample["target"]:
                if isinstance(sample["target"][key], torch.Tensor):
                    sample["target"][key] = sample["target"][key].to(device)

        # polynomial_decay LR step
        if args.lr_scheduler == "polynomial_decay":
            global_step = (epoch - 1) * len(train_loader) + i
            new_lr = _lr_polynomial_decay(
                args.lr, global_step, total_steps, warmup_steps, power=args.lr_power)
            for pg in optimizer.param_groups:
                pg["lr"] = new_lr

        optimizer.zero_grad()

        if args.fp16 and device.type == "cuda":
            with autocast():
                loss, sample_size, logging_output = criterion(model, sample, reduce=True)
            loss = loss / sample_size
            scaler.scale(loss).backward()
            if args.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, sample_size, logging_output = criterion(model, sample, reduce=True)
            loss = loss / sample_size
            loss.backward()
            if args.clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

        total_loss += logging_output["loss"].item()
        total_samples += sample_size
        avg_loss = total_loss / max(total_samples, 1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(task, model, criterion, valid_loader, device, args):
    model.eval()
    criterion.eval()
    all_predicts = []
    all_targets = []
    total_loss = 0.0
    total_samples = 0

    for sample in valid_loader:
        for key in sample.get("net_input", {}):
            if isinstance(sample["net_input"][key], torch.Tensor):
                sample["net_input"][key] = sample["net_input"][key].to(device)
        if "target" in sample:
            for key in sample["target"]:
                if isinstance(sample["target"][key], torch.Tensor):
                    sample["target"][key] = sample["target"][key].to(device)

        if args.fp16 and device.type == "cuda":
            with autocast():
                loss, sample_size, logging_output = criterion(model, sample, reduce=True)
        else:
            loss, sample_size, logging_output = criterion(model, sample, reduce=True)

        total_loss += logging_output["loss"].item()
        total_samples += sample_size

        if "predict" in logging_output:
            all_predicts.append(logging_output["predict"].cpu().numpy())
            all_targets.append(logging_output["target"].cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    metrics = {"loss": avg_loss}

    if all_predicts:
        y_pred = np.concatenate(all_predicts, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        # loss 已在 eval 模式下反归一化 predict，target 是 LMDB 原始值，不再重复处理

        # per-target R² and MAE
        num_targets = y_pred.shape[1] if y_pred.ndim > 1 else 1
        target_names = getattr(args, "target_names", None) or \
            task.args.__dict__.get("target_names", None)

        maes = []
        r2s = []
        for j in range(num_targets):
            yp = y_pred[:, j] if num_targets > 1 else y_pred
            yt = y_true[:, j] if num_targets > 1 else y_true
            mae_j = float(np.abs(yp - yt).mean())
            r2_j = float(r2_score(yt, yp))
            maes.append(mae_j)
            r2s.append(r2_j)
            name = target_names[j] if target_names and j < len(target_names) else f"target_{j}"
            logger.info("  %-12s  MAE %8.4f  R² %7.4f", name, mae_j, r2_j)
            metrics[f"mae_{name}"] = mae_j
            metrics[f"r2_{name}"] = r2_j

        # aggregate metrics
        agg_mae = np.mean(maes)
        agg_r2 = np.mean(r2s)
        metrics["mae"] = agg_mae
        metrics["r2"] = agg_r2
        metrics["valid_agg_mae"] = agg_mae
        metrics["valid_agg_r2"] = agg_r2
        logger.info("  %-12s  MAE %8.4f  R² %7.4f", "AGGREGATE", agg_mae, agg_r2)
    else:
        logger.info("validation loss %.4f", avg_loss)

    return metrics


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = build_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    if args.fp16 and device.type != "cuda":
        logger.info("FP16 disabled — no CUDA")
        args.fp16 = False

    os.makedirs(args.save_dir, exist_ok=True)

    # task / model / criterion
    task = setup_task(args)
    task.load_dataset("train")
    task.load_dataset("valid")

    model = task.build_model(args)
    model = model.to(device)

    # freeze encoder layers
    if args.freeze_encoder_layers > 0:
        frozen = 0
        for name, param in model.named_parameters():
            # 参数名形如 encoders.0.layers.<i>.<...>（Uni-Mol 用 encoders 复数 + 索引）
            if "encoders." in name and ".layers." in name:
                try:
                    layer_idx = int(name.split(".layers.")[1].split(".")[0])
                    if layer_idx < args.freeze_encoder_layers:
                        param.requires_grad = False
                        frozen += 1
                except (ValueError, IndexError):
                    pass
        logger.info("Frozen %d params in encoder layers 0-%d", frozen, args.freeze_encoder_layers - 1)

    criterion = task.build_criterion(args)
    criterion = criterion.to(device)

    # optimizer
    betas = parse_betas(args.adam_betas)
    optimizer = Adam(model.parameters(), lr=args.lr, betas=betas, eps=args.adam_eps,
                     weight_decay=args.weight_decay)

    # fp16 scaler
    scaler = GradScaler(init_scale=args.fp16_init_scale,
                        growth_interval=args.fp16_scale_window) if args.fp16 else None

    # data loaders
    train_loader = task.get_batch_iterator(
        task.datasets["train"], batch_size=args.batch_size,
        shuffle=not args.no_shuffle, num_workers=args.num_workers)
    valid_loader = task.get_batch_iterator(
        task.datasets["valid"], batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers)

    # LR scheduler constants
    total_steps = args.max_epoch * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    logger.info("Total steps: %d, warmup: %d (%.1f%%)", total_steps, warmup_steps, args.warmup_ratio * 100)

    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.max_epoch + 1):
        lr = optimizer.param_groups[0]["lr"]
        logger.info("Epoch %d/%d  lr=%.2e  patience=%d/%d", epoch, args.max_epoch, lr,
                    patience_counter, args.patience)
        logger.info("-" * 50)

        train_loss = train_epoch(epoch, task, model, criterion, optimizer, scaler,
                                 train_loader, device, total_steps, warmup_steps, args)
        logger.info("epoch %d train loss: %.4f", epoch, train_loss)

        if epoch % args.validate_interval == 0:
            metrics = validate(task, model, criterion, valid_loader, device, args)
            current = metrics.get(args.best_checkpoint_metric, metrics.get("mae", float("inf")))

            if current < best_metric:
                best_metric = current
                best_epoch = epoch
                patience_counter = 0
                path = os.path.join(args.save_dir, "checkpoint_best.pt")
                checkpoint_utils.save_checkpoint(path, {
                    "args": args, "model": model.state_dict(),
                    "epoch": epoch, "metrics": metrics,
                })
                logger.info("saved best checkpoint to %s (metric=%.4f)", path, current)
            else:
                patience_counter += 1

            if args.patience > 0 and patience_counter >= args.patience:
                logger.info("Early stopping at epoch %d (best epoch=%d, metric=%.4f)",
                           epoch, best_epoch, best_metric)
                break

    logger.info("Training complete. Best %s=%.4f at epoch %d",
                args.best_checkpoint_metric, best_metric, best_epoch)


if __name__ == "__main__":
    main()
