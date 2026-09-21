#!/usr/bin/env python3
"""Extract UniMol CLS embeddings from an LMDB dataset without unicore.

This is a no-unicore reimplementation of the embedding extraction logic in
/mnt/f/green-laser/feature_transfer.py.
"""

import argparse
import os
import pickle
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import torch
import lmdb
from tqdm import tqdm

from unicore_compat.checkpoint_utils import load_checkpoint_to_cpu
from unicore_compat.data import Dictionary
from unimol.models.unimol_test import UniMolModel


def build_args(state):
    from argparse import Namespace
    args = Namespace()
    ckpt_args = state.get("args", None)
    if ckpt_args is not None:
        for k, v in vars(ckpt_args).items():
            setattr(args, k, v)
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
        "num_classes": 2,          # 预训练 checkpoint 的 args 里没有，构造 UniMolModel 时需要
        "mode": "infer",
    }
    for k, v in defaults.items():
        if not hasattr(args, k):
            setattr(args, k, v)
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


@torch.no_grad()
def extract_embeddings(model, dictionary, lmdb_path, batch_size=16, layers=None):
    """提取 CLS 表征。layers: 1-based 层号列表（None = 最后一层）。
    返回 (dict{层号: (n,d) 数组}, targets, names)。"""
    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(10e9),
    )
    txn = env.begin()
    keys = list(txn.cursor().iternext(values=False))
    n = len(keys)

    device = next(model.parameters()).device
    n_tok = len(dictionary)
    embed_dim = model.args.encoder_embed_dim
    _layers = list(layers) if layers else [model.args.encoder_layers]
    all_embeddings = {L: [] for L in _layers}
    all_targets = []
    all_names = []

    for start in tqdm(range(0, n, batch_size), desc=f"Extract {os.path.basename(lmdb_path)}"):
        end = min(start + batch_size, n)
        batch_atoms, batch_coords, batch_targets, batch_names = [], [], [], []
        for idx in range(start, end):
            dp = txn.get(f"{idx}".encode("ascii"))
            if dp is None:
                continue
            obj = pickle.loads(dp)
            atoms = obj["atoms"][0] if isinstance(obj["atoms"], list) else obj["atoms"]
            coords = (
                obj["coordinates"][0]
                if isinstance(obj["coordinates"], list) and len(obj["coordinates"]) > 0
                else obj["coordinates"]
            )
            target = obj.get("target", [0.0, 0.0])
            name = obj.get("name", f"mol_{idx}")
            batch_atoms.append(atoms)
            batch_coords.append(coords)
            batch_targets.append(target)
            batch_names.append(name)

        if len(batch_atoms) == 0:
            continue

        bs = len(batch_atoms)
        max_len = max(len(a) for a in batch_atoms)
        src_tokens = torch.zeros(bs, max_len, dtype=torch.long, device=device)
        src_coord = torch.zeros(bs, max_len, 3, dtype=torch.float, device=device)

        for i, atoms in enumerate(batch_atoms):
            n_a = len(atoms)
            tokens = [dictionary.index(str(a)) for a in atoms]
            src_tokens[i, :n_a] = torch.tensor(tokens, device=device)
            src_tokens[i, 0] = dictionary.bos()
            src_coord[i, :n_a] = torch.tensor(np.array(batch_coords[i]), device=device)

        src_distance = torch.cdist(src_coord, src_coord)
        src_edge_type = src_tokens.unsqueeze(-1) * n_tok + src_tokens.unsqueeze(-2)

        captured = {}

        def _mk(L):
            def hook(module, input, output):
                captured[L] = output[0][:, 0, :]
            return hook

        hooks = [model.encoders[0].layers[L - 1].register_forward_hook(_mk(L)) for L in _layers]
        model(src_tokens, src_distance, src_coord, src_edge_type)
        for h in hooks:
            h.remove()

        n_got = min(captured[_layers[0]].shape[0], len(batch_names))
        for L in _layers:
            all_embeddings[L].append(captured[L].cpu().numpy()[:n_got])
        all_targets.extend(batch_targets[:n_got])
        all_names.extend(batch_names[:n_got])

    env.close()
    return (
        {L: np.concatenate(v, axis=0) for L, v in all_embeddings.items()},
        np.array(all_targets, dtype=np.float32),
        all_names,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to UniMol .pt checkpoint")
    parser.add_argument("--dict", dest="dict_path", required=True, help="Path to dictionary file")
    parser.add_argument("--lmdb", required=True, help="Path to input LMDB file")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--layers", default="",
                        help="要提取的层(1-based)，逗号分隔，如 5,10,15；留空=最后一层")
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    layers = [int(x) for x in args.layers.split(",") if x.strip()] if args.layers else None
    model, dictionary, margs = load_model(args.checkpoint, args.dict_path, device)
    if layers:
        bad = [L for L in layers if L < 1 or L > margs.encoder_layers]
        if bad:
            raise SystemExit(f"层号超出范围 1..{margs.encoder_layers}: {bad}")
    embs, targets, names = extract_embeddings(
        model, dictionary, args.lmdb, batch_size=args.batch_size, layers=layers
    )
    payload = {f"L{L}": v for L, v in embs.items()}
    payload["embeddings"] = embs[max(embs)]          # 兼容旧调用方：最后一层
    payload["targets"] = targets
    payload["names"] = np.array(names)
    np.savez(args.output, **payload)
    print(f"Saved {len(names)} embeddings to {args.output} 层: {sorted(embs)} "
          f"每层维度: {[v.shape[1] for _, v in sorted(embs.items())]}")


if __name__ == "__main__":
    main()
