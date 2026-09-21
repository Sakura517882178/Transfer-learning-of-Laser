#!/usr/bin/env python3
"""Native PyTorch pretraining loop for UniMol without unicore."""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
from torch.optim import Adam

from unicore_compat import checkpoint_utils
from unicore_compat.tasks import setup_task

import unimol.tasks.unimol  # register unimol task
import unimol.models.unimol  # register unimol_ori model
import unimol.losses.unimol  # register unimol loss

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_parser():
    parser = argparse.ArgumentParser("Pretrain UniMol without unicore")
    parser.add_argument("data", help="path to data directory containing train.lmdb / valid.lmdb")
    parser.add_argument("--task", default="unimol")
    parser.add_argument("--loss", default="unimol")
    parser.add_argument("--arch", default="unimol_ori")
    parser.add_argument("--dict-name", default="pretrain_dict.txt")
    parser.add_argument("--encoder-layers", type=int, default=15)
    parser.add_argument("--encoder-embed-dim", type=int, default=512)
    parser.add_argument("--encoder-ffn-embed-dim", type=int, default=2048)
    parser.add_argument("--encoder-attention-heads", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--attention-dropout", type=float, default=0.1)
    parser.add_argument("--activation-dropout", type=float, default=0.0)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--max-atoms", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-epoch", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-dir", default="./pretrain_checkpoints")
    parser.add_argument("--finetune-encoder-model", default=None)
    parser.add_argument("--pretrain-path", default=None)
    parser.add_argument("--mask-prob", type=float, default=0.15)
    parser.add_argument("--leave-unmasked-prob", type=float, default=0.05)
    parser.add_argument("--random-token-prob", type=float, default=0.05)
    parser.add_argument("--noise-type", default="uniform")
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--remove-hydrogen", action="store_true")
    parser.add_argument("--remove-polar-hydrogen", action="store_true")
    parser.add_argument("--only-polar", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--valid-interval", type=int, default=1)
    # Pretraining loss weights
    parser.add_argument("--masked-token-loss", type=float, default=1.0)
    parser.add_argument("--masked-coord-loss", type=float, default=1.0)
    parser.add_argument("--masked-dist-loss", type=float, default=1.0)
    parser.add_argument("--x-norm-loss", type=float, default=0.0)
    parser.add_argument("--delta-pair-repr-norm-loss", type=float, default=-1.0)
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
        "num_classes": 1,
        "mode": "train",
        "conf_size": 1,
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
    return args


def train_epoch(epoch, task, model, criterion, optimizer, train_loader, device):
    model.train()
    criterion.train()
    task.args.epoch = epoch
    if hasattr(train_loader.dataset, "set_epoch"):
        train_loader.dataset.set_epoch(epoch)

    total_loss = 0.0
    n_batches = 0
    for i, sample in enumerate(train_loader):
        for key in sample.get("net_input", {}):
            if isinstance(sample["net_input"][key], torch.Tensor):
                sample["net_input"][key] = sample["net_input"][key].to(device)
        if "target" in sample:
            for key in sample["target"]:
                if isinstance(sample["target"][key], torch.Tensor):
                    sample["target"][key] = sample["target"][key].to(device)

        optimizer.zero_grad()
        loss, _, logging_output = criterion(model, sample, reduce=True)
        loss.backward()
        optimizer.step()

        total_loss += logging_output["loss"].item()
        n_batches += 1

        if (i + 1) % task.args.log_interval == 0:
            logger.info(
                "epoch %d | batch %d/%d | loss %.4f",
                epoch,
                i + 1,
                len(train_loader),
                total_loss / max(n_batches, 1),
            )

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(task, model, criterion, valid_loader, device):
    model.eval()
    criterion.eval()
    total_loss = 0.0
    n_batches = 0
    for sample in valid_loader:
        for key in sample.get("net_input", {}):
            if isinstance(sample["net_input"][key], torch.Tensor):
                sample["net_input"][key] = sample["net_input"][key].to(device)
        if "target" in sample:
            for key in sample["target"]:
                if isinstance(sample["target"][key], torch.Tensor):
                    sample["target"][key] = sample["target"][key].to(device)

        loss, _, logging_output = criterion(model, sample, reduce=True)
        total_loss += logging_output["loss"].item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    logger.info("validation loss %.4f", avg_loss)
    return {"loss": avg_loss}


def main():
    parser = get_parser()
    args = parser.parse_args()
    args = build_args(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    os.makedirs(args.save_dir, exist_ok=True)

    task = setup_task(args)
    task.load_dataset("train")
    task.load_dataset("valid")

    model = task.build_model(args)
    model = model.to(device)
    criterion = task.build_criterion(args)
    criterion = criterion.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)

    train_loader = task.get_batch_iterator(
        task.datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = task.get_batch_iterator(
        task.datasets["valid"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    best_loss = float("inf")
    for epoch in range(1, args.max_epoch + 1):
        logger.info("=" * 60)
        logger.info("Epoch %d/%d", epoch, args.max_epoch)
        train_loss = train_epoch(
            epoch, task, model, criterion, optimizer, train_loader, device
        )
        logger.info("epoch %d train loss: %.4f", epoch, train_loss)

        if epoch % args.valid_interval == 0:
            metrics = validate(task, model, criterion, valid_loader, device)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                path = os.path.join(args.save_dir, "checkpoint_best.pt")
                checkpoint_utils.save_checkpoint(
                    path,
                    {
                        "args": args,
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "metrics": metrics,
                    },
                )
                logger.info("saved best checkpoint to %s", path)

    logger.info("Pretraining complete. Best validation loss: %.4f", best_loss)


if __name__ == "__main__":
    main()
