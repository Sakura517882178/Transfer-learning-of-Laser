#!/usr/bin/env python3
"""正文图 3（按原图 3 的四面板结构，改为等训练预算口径）：
 (a) t-SNE：源域 1010 vs 目标域 69（源域 Uni-Mol 编码器 CLS 第 10 层）
 (b) n = 6 的逐 epoch 曲线（S1 MAE / σopt MAE / S1 R² / σopt R²；迁移实线 vs 随机初始化虚线，10 组划分 mean±SD）
 (c) n = 53 同上
 (d) 4 个指标在 20 epoch 预算下随训练规模的柱状对比（两臂，误差棒 = 10 组划分 SD）
输出：E:\paper-lasergreen\version1\figures\Fig3_transfer_learning.{png,pdf}
"""
import glob, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

W = "/mnt/f/green-laser/3core"
REP = f"{W}/report"
OUT = "/mnt/e/paper-lasergreen/version1/figures"
SIZES = [6, 13, 27, 40, 53]
BUDGET = 20
C_TR, C_BL = "#1F4E79", "#4E9F9B"      # 迁移=深蓝实线 / 基线=青绿虚线（同原图 3 观感）
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 1.0,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
                     "axes.labelsize": 9.5, "savefig.bbox": "tight", "figure.dpi": 130})

EP = re.compile(r"epoch (\d+) train loss")
M0 = re.compile(r"target_0\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
M1 = re.compile(r"target_1\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
AG = re.compile(r"AGGREGATE\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")


def load(path):
    d, cur = {}, None
    for line in open(path, errors="ignore").read().replace("\r", "\n").split("\n"):
        m = EP.search(line)
        if m:
            cur = int(m.group(1)); d.setdefault(cur, {})
        if cur is None:
            continue
        a, b, c = M0.search(line), M1.search(line), AG.search(line)
        if a:
            d[cur]["s1mae"], d[cur]["s1r2"] = float(a.group(1)), float(a.group(2))
        if b:
            d[cur]["sgmae"], d[cur]["sgr2"] = float(b.group(1)), float(b.group(2))
        if c:
            d[cur]["agg"], d[cur]["tot"] = float(c.group(1)), float(c.group(2))
    return d


runs = {}
for n in SIZES:
    for s in range(10):
        for arm in ("transfer", "baseline"):
            p = f"{W}/runs/repeat2/tr{n}_s{s:02d}/{arm}/save_finetune/train.log"
            if os.path.exists(p):
                runs[f"{n}|{s}|{arm}"] = load(p)


def band(n_, arm, key, upto=60):
    curves = []
    for s in range(10):
        d = runs.get(f"{n_}|{s}|{arm}", {})
        eps = sorted(e for e in d if e <= upto and key in d[e])
        curves.append([d[e][key] for e in eps])
    L = min(len(x) for x in curves) if curves else 0
    if not L:
        return None, None, None
    A = np.array([x[:L] for x in curves])
    return np.arange(1, L + 1), A.mean(0), A.std(0, ddof=1)


fig = plt.figure(figsize=(13.8, 11.4))
gs = GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.42)

# ── (a) t-SNE ──
zs, z0, z1 = (np.load(f"{REP}/align/{f}") for f in ("source_0.npz", "target_0.npz", "target_1.npz"))
Xs = zs["L10"].astype(float)
Xt = np.vstack([z0["L10"], z1["L10"]]).astype(float)
tn = [x.replace("_valid", "").replace("_train", "") for x in list(z0["names"]) + list(z1["names"])]
Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(
    PCA(50).fit_transform(np.vstack([Xs, Xt])))
axa = fig.add_subplot(gs[0:2, 0:2])
axa.scatter(Z[:len(Xs), 0], Z[:len(Xs), 1], s=7, c="#AFC6DA", alpha=.6, lw=0,
            label=f"Source (w/ C=C, n={len(Xs)})")
axa.scatter(Z[len(Xs):, 0], Z[len(Xs):, 1], s=30, c="#D07A2F", alpha=.9, lw=.3, ec="w",
            label=f"Target (w/o C=C, n={len(Xt)})")
for i, nm in enumerate(tn):
    if nm in ("Th_Th_Fl", "Py_dTT_Fl"):
        axa.annotate("FL3T" if nm == "Th_Th_Fl" else "PyDTTFL", (Z[len(Xs) + i, 0], Z[len(Xs) + i, 1]),
                     xytext=(5, -10), textcoords="offset points", fontsize=8, color="#8C4A0F")
axa.set_xlabel("t-SNE 1"); axa.set_ylabel("t-SNE 2")
axa.legend(loc="upper right", fontsize=8.5)
axa.text(-0.12, 1.06, "(a)", transform=axa.transAxes, fontsize=13)

# ── (b)(c) 逐 epoch 曲线 ──
SPEC = [("s1mae", "S$_1$ MAE (eV)", "S$_1$ Error"), ("sgmae", r"$\sigma_{\mathrm{opt}}$ MAE", r"$\sigma_{\mathrm{opt}}$ Error"),
        ("s1r2", "S$_1$ R$^2$", "S$_1$ Accuracy"), ("sgr2", r"$\sigma_{\mathrm{opt}}$ R$^2$", r"$\sigma_{\mathrm{opt}}$ Accuracy")]
for tag, (n_, row0, col0) in (("(b)", (6, 0, 2)), ("(c)", (53, 2, 0))):
    first = None
    for j, (key, ylab, ttl) in enumerate(SPEC):
        ax = fig.add_subplot(gs[row0 + j // 2, col0 + j % 2])
        if first is None:
            first = ax
        for arm, c, ls, nm in (("transfer", C_TR, "-", f"Transfer (n = {n_})"),
                               ("baseline", C_BL, "--", f"Random-init (n = {n_})")):
            e, m, sd = band(n_, arm, key)
            if e is None:
                continue
            ax.plot(e, m, color=c, lw=1.3, ls=ls, label=nm)
            ax.fill_between(e, m - sd, m + sd, color=c, alpha=.16, lw=0)
        ax.axvline(BUDGET, color="gray", lw=.7, ls=":")
        ax.set_title(ttl, fontsize=9.5)
        ax.set_xlabel("Epoch", fontsize=8.5); ax.set_ylabel(ylab, fontsize=8.5)
        ax.set_xlim(1, 40); ax.tick_params(labelsize=8)
        if j == 0:
            ax.legend(fontsize=8, loc="upper right")
    first.text(-0.32, 1.30, tag, transform=first.transAxes, fontsize=13)

# ── (d) 20 epoch 预算下的柱状对比（4 指标 × 5 规模 × 2 臂）──
BARS = [("agg", "Aggregate MAE", "Aggregate MAE"), ("s1r2", "S$_1$ R$^2$", "S$_1$ Accuracy"),
        ("sgr2", r"$\sigma_{\mathrm{opt}}$ R$^2$", r"$\sigma_{\mathrm{opt}}$ Accuracy"),
        ("tot", "Total R$^2$", "Total Accuracy")]
first = None
for j, (key, ylab, ttl) in enumerate(BARS):
    ax = fig.add_subplot(gs[2 + j // 2, 2 + j % 2])
    if first is None:
        first = ax
    x = np.arange(len(SIZES)); w = 0.36
    for k, (arm, c, nm) in enumerate((("transfer", C_TR, "Transfer"), ("baseline", C_BL, "Random-init"))):
        vals, errs = [], []
        for n_ in SIZES:
            v = [runs[f"{n_}|{s}|{arm}"].get(BUDGET, {}).get(key) for s in range(10)]
            v = [z for z in v if z is not None]
            vals.append(np.mean(v)); errs.append(np.std(v, ddof=1))
        ax.bar(x + (k - .5) * w, vals, w, yerr=errs, capsize=2, color=c, edgecolor="k", lw=.6,
               label=nm, error_kw=dict(lw=.8))
    ax.set_xticks(x); ax.set_xticklabels(SIZES, fontsize=8.5)
    ax.set_xlabel(f"Training molecules ({BUDGET}-epoch budget)", fontsize=8.5)
    ax.set_ylabel(ylab, fontsize=8.5)
    ax.set_title(ttl, fontsize=9.5); ax.tick_params(labelsize=8)
    if j == 0:
        ax.legend(fontsize=8, loc="upper right")
first.text(-0.30, 1.30, "(d)", transform=first.transAxes, fontsize=13)

for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig3_transfer_learning.{ext}", dpi=600)
plt.close(fig)
print(f"→ {OUT}/Fig3_transfer_learning.png")
