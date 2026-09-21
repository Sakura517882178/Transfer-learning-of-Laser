#!/usr/bin/env python3
"""Fig 3：仿照原文图 3 的四面板结构与内容，用新的 10 组划分数据重画
 (a) t-SNE：源域 1010 分子 vs 目标域 69 分子（Uni-Mol 编码器 CLS，第 10 层）
 (b) n = 6 的逐 epoch 曲线（4 指标：S1 MAE / σopt MAE / S1 R² / σopt R²，两臂）
 (c) n = 53 的逐 epoch 曲线（同上）
 (d) 5 个训练规模的 4 个指标柱状对比（Aggregate MAE / S1 R² / σopt R² / Total R²，误差棒=10 组划分 SD）
"""
import csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

OUT = "/mnt/e/paper-lasergreen/version1/figures"
RUNS = "/mnt/f/green-laser/3core/runs/repeat"
REP = "/mnt/f/green-laser/3core/report"
SIZES = [6, 13, 27, 40, 53]
C_TR, C_BL = "#1F4E79", "#4E9F9B"      # 迁移=深蓝(实线)，基线=青绿(虚线) —— 与原文图 3 一致的观感
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
                     "axes.labelsize": 9.5, "savefig.bbox": "tight", "figure.dpi": 130})
EP = re.compile(r"epoch (\d+) train loss")
M0 = re.compile(r"target_0\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
M1 = re.compile(r"target_1\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
AG = re.compile(r"AGGREGATE\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")


def style(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.0)


def parse(log):
    rows, cur = [], {}
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        m = EP.search(line)
        if m:
            cur = {"epoch": int(m.group(1))}
        a, b, c = M0.search(line), M1.search(line), AG.search(line)
        if a and cur:
            cur["s1m"], cur["s1r"] = float(a.group(1)), float(a.group(2))
        if b and cur:
            cur["sgm"], cur["sgr"] = float(b.group(1)), float(b.group(2))
        if c and cur:
            cur["agg"], cur["tot"] = float(c.group(1)), float(c.group(2)); rows.append(cur); cur = {}
    return rows


per = json.load(open(f"{REP}/repeat5_per_run.json"))

fig = plt.figure(figsize=(13.6, 11.2))
gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)

# ── (a) t-SNE ──
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
zs, z0, z1 = (np.load(f"{REP}/align/{f}") for f in ("source_0.npz", "target_0.npz", "target_1.npz"))
Xs, Xt = zs["L10"].astype(float), np.vstack([z0["L10"], z1["L10"]]).astype(float)
tn = [n.replace("_valid", "").replace("_train", "") for n in list(z0["names"]) + list(z1["names"])]
X = np.vstack([Xs, Xt])
Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(PCA(50).fit_transform(X))
axa = fig.add_subplot(gs[0:2, 0:2])
axa.scatter(Z[:len(Xs), 0], Z[:len(Xs), 1], s=7, c="#AFC6DA", alpha=.6, lw=0, label=f"Source (w/ C=C, n={len(Xs)})")
axa.scatter(Z[len(Xs):, 0], Z[len(Xs):, 1], s=30, c="#D07A2F", alpha=.9, lw=.3, ec="w",
            label=f"Target (w/o C=C, n={len(Xt)})")
for i, nm in enumerate(tn):
    if nm in ("Th_Th_Fl", "Py_dTT_Fl"):
        axa.annotate("FL3T" if nm == "Th_Th_Fl" else "PyDTTFL", (Z[len(Xs) + i, 0], Z[len(Xs) + i, 1]),
                     xytext=(5, -10), textcoords="offset points", fontsize=8, color="#8C4A0F")
axa.set_xlabel("t-SNE 1"); axa.set_ylabel("t-SNE 2"); style(axa)
axa.legend(loc="upper right", fontsize=8.5)
axa.text(-0.12, 1.06, "(a)", transform=axa.transAxes, fontsize=13)

# ── (b)/(c) 逐 epoch 曲线（4 指标）──
SPEC = [("s1m", "S$_1$ MAE (eV)", "S$_1$ Error"), ("sgm", r"$\sigma_{\mathrm{opt}}$ MAE (10$^{-16}$ cm$^2$)", r"$\sigma_{\mathrm{opt}}$ Error"),
        ("s1r", "S$_1$ R$^2$", "S$_1$ Accuracy"), ("sgr", r"$\sigma_{\mathrm{opt}}$ R$^2$", r"$\sigma_{\mathrm{opt}}$ Accuracy")]
for k, (n, row0, col0, tag) in enumerate(((6, 0, 2, "(b)"), (53, 2, 0, "(c)"))):
    first = None
    for j, (key, ylab, ttl) in enumerate(SPEC):
        ax = fig.add_subplot(gs[row0 + j // 2, col0 + j % 2])
        if first is None:
            first = ax
        for arm, c, nm, ls in (("transfer", C_TR, f"Transfer (n = {n})", "-"), ("baseline", C_BL, f"Random-init baseline (n = {n})", "--")):
            lg = f"{RUNS}/tr{n}_s00/{arm}/save_finetune/train.log"
            if not os.path.exists(lg):
                continue
            rows = [r for r in parse(lg) if key in r]
            ax.plot([r["epoch"] for r in rows], [r[key] for r in rows], color=c, lw=1.2, ls=ls, label=nm)
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8.5); ax.set_ylabel(ylab, fontsize=8.5)
        ax.tick_params(labelsize=8); style(ax)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    first.text(-0.30, 1.28, tag, transform=first.transAxes, fontsize=13)

# ── (d) 柱状对比（4 指标 × 5 规模 × 2 臂，误差棒 = 10 组划分 SD）──
BARS = [("agg", "Aggregate MAE", "Aggregate MAE"), ("s1_r2", "S$_1$ R$^2$", "S$_1$ Accuracy"),
        ("sg_r2", r"$\sigma_{\mathrm{opt}}$ R$^2$", r"$\sigma_{\mathrm{opt}}$ Accuracy"),
        ("tot_r2", "Total R$^2$", "Total Accuracy")]
first = None
for j, (key, ylab, ttl) in enumerate(BARS):
    ax = fig.add_subplot(gs[2 + j // 2, 2 + j % 2])
    if first is None:
        first = ax
    x = np.arange(len(SIZES)); w = 0.36
    for k, (arm, c, nm) in enumerate((("transfer", C_TR, "Transfer"), ("baseline", C_BL, "Random-init baseline"))):
        m = [np.mean([per[f"{n}|{s}|{arm}"][key] for s in range(10)]) for n in SIZES]
        e = [np.std([per[f"{n}|{s}|{arm}"][key] for s in range(10)], ddof=1) for n in SIZES]
        ax.bar(x + (k - .5) * w, m, w, yerr=e, capsize=2, color=c, edgecolor="k", lw=.6, label=nm,
               error_kw=dict(lw=.8))
    ax.set_xticks(x); ax.set_xticklabels(SIZES, fontsize=8.5)
    ax.set_xlabel("Training molecules", fontsize=8.5); ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(ttl, fontsize=9); ax.tick_params(labelsize=8); style(ax)
    if j == 0:
        ax.legend(fontsize=8, loc="upper right")
first.text(-0.30, 1.28, "(d)", transform=first.transAxes, fontsize=13)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig3_transfer_learning.{ext}", dpi=600)
plt.close(fig)
print("→", f"{OUT}/Fig3_transfer_learning.png")
