#!/usr/bin/env python3
"""Minimal no-unicore UniMol inference script.

Loads a UniMol checkpoint that was trained with the unicore backend and runs a
single forward pass using only PyTorch + this no-unicore fork.
"""

import argparse
import os
import sys

# Add the repository root so that `import unimol` and `import unicore_compat` work.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch

from unicore_compat.checkpoint_utils import load_checkpoint_to_cpu
from unicore_compat.data import Dictionary
from unimol.models.unimol_test import UniMolModel


def build_args(checkpoint_state):
    """Build an argparse.Namespace from checkpoint args, filling no-unicore defaults."""
    ckpt_args = checkpoint_state.get("args", None)
    args = argparse.Namespace()

    # Copy over every attribute stored in the checkpoint args.
    if ckpt_args is not None:
        for key, value in vars(ckpt_args).items():
            setattr(args, key, value)

    # Ensure the architecture defaults are present.
    defaults = {
        "encoder_layers": 15,
        "encoder_embed_dim": 512,
        "encoder_ffn_embed_dim": 2048,
        "encoder_attention_heads": 64,
        "dropout": 0.1,
        "emb_dropout": 0.1,
        "attention_dropout": 0.1,
        "activation_dropout": 0.0,
        "pooler_dropout": 0.0,
        "max_seq_len": 512,
        "activation_fn": "gelu",
        "pooler_activation_fn": "tanh",
        "post_ln": False,
        "masked_token_loss": -1.0,
        "masked_coord_loss": -1.0,
        "masked_dist_loss": -1.0,
        "x_norm_loss": -1.0,
        "delta_pair_repr_norm_loss": -1.0,
        "mode": "infer",
    }
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Force inference mode regardless of how the model was trained.
    args.mode = "infer"
    return args


def load_model(weights_path, dict_path, device="cpu"):
    dictionary = Dictionary.load(dict_path)
    dictionary.add_symbol("[MASK]", is_special=True)
    state = load_checkpoint_to_cpu(weights_path)
    args = build_args(state)
    model = UniMolModel(args, dictionary)
    model.load_state_dict(state["model"], strict=False)
    model = model.to(device)
    model.eval()
    return model, dictionary, args


def make_dummy_input(num_atoms, dictionary, device="cpu"):
    """Create a single-molecule dummy input for a quick forward pass."""
    batch_size = 1
    # Use real atom tokens from the dictionary, skipping pad/eos/unk for the body.
    valid_tokens = [i for i in range(len(dictionary)) if i not in (
        dictionary.pad(), dictionary.eos(), dictionary.unk()
    )]
    src_tokens = torch.tensor(
        [[dictionary.pad()] + valid_tokens[:num_atoms - 1]], device=device
    )
    n = src_tokens.size(1)
    src_distance = torch.rand(batch_size, n, n, device=device)
    src_coord = torch.rand(batch_size, n, 3, device=device)
    src_edge_type = torch.randint(
        0, len(dictionary) * len(dictionary), (batch_size, n, n), device=device
    )
    return {
        "src_tokens": src_tokens,
        "src_distance": src_distance,
        "src_coord": src_coord,
        "src_edge_type": src_edge_type,
    }


def main():
    parser = argparse.ArgumentParser(description="Run UniMol inference without unicore")
    parser.add_argument("--weights", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--dict", dest="dict_path", required=True, help="Path to dictionary file")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--num-atoms", type=int, default=10, help="Number of atoms in dummy input")
    args = parser.parse_args()

    print("Loading model...")
    model, dictionary, model_args = load_model(args.weights, args.dict_path, args.device)
    print(f"Model loaded. Dictionary size: {len(dictionary)}, num_classes: {model_args.num_classes}")

    sample = make_dummy_input(args.num_atoms, dictionary, args.device)
    with torch.no_grad():
        encoder_rep, encoder_pair_rep = model(**sample)

    print("Forward pass succeeded.")
    print("  encoder_rep shape:", encoder_rep.shape)
    print("  encoder_pair_rep shape:", encoder_pair_rep.shape)


if __name__ == "__main__":
    main()
