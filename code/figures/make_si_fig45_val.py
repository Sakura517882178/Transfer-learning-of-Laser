#!/usr/bin/env python3
"""SI 图 S4 / S5 重做（基于留出验证集，不含训练分子）：
  S4：σopt 排序相关系数随训练规模的变化（迁移 vs 基线，10 组划分均值±SD）
  S5：σopt 分档平均绝对误差（按参考值四分位分档；迁移 vs 基线）
数据：report/valrank/*.csv（100 个 run 的验证集逐样本预测，z-score → 用各 run 的 norms 还原物理量）
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

W = "/mnt/f/green-laser/3core"
VAL = f"{W}/report/valrank"
OUT = "/mnt/e/paper-lasergreen/version2/figures"
os.makedirs(OUT, exist_ok=True)
SIZES = [6, 13, 27, 40, 53]
plt.rcParams.update({"font.size": 10.5, "font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "xtick.direction": "out", "ytick.direction": "out", "figure.dpi": 130,
                     "savefig.bbox": "tight"})
C_T, C_B = "#1F4E79", "#B4553C"


def norms(sid, arm):
    p = f"{W}/runs/repeat2/{sid}/{arm}/property_norms.json"
    d = json.load(open(p))["task_data"]
    return np.array(d["mean"]), np.array(d["std"])


def load(f):
    sid, arm = os.path.basename(f)[:-4].rsplit("_", 1)
    rows = list(csv.DictReader(open(f)))
    m, s = norms(sid, arm)
    s1t = np.array([float(r["target_0_true"]) for r in rows])
    s1p = np.array([float(r["target_0_pred"]) for r in rows]) * s[0] + m[0]
    sgt = np.array([float(r["target_1_true"]) for r in rows])
    sgp = np.array([float(r["target_1_pred"]) for r in rows]) * s[1] + m[1]
    return s1t, s1p, sgt, sgp


# ---------- 图 S4：排序相关系数 vs 训练规模 ----------
fig, ax = plt.subplots(figsize=(6.0, 4.2))
for arm, col, lab in (("transfer", C_T, "Transfer learning"), ("baseline", C_B, "Randomly initialized baseline")):
    mu, sd = [], []
    for n in SIZES:
        v = [spearmanr(*load(f)[2:])[0] for f in sorted(glob.glob(f"{VAL}/tr{n}_s*_{arm}.csv"))]
        mu.append(np.mean(v)); sd.append(np.std(v, ddof=1))
    ax.errorbar(SIZES, mu, yerr=sd, marker="o", ms=5, lw=1.8, capsize=3.5, color=col, label=lab)
    for x, m in zip(SIZES, mu):
        ax.annotate(f"{m:.2f}", (x, m), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("Number of training molecules")
ax.set_ylabel(r"Spearman $\rho$ of $\sigma_{\mathrm{opt}}$ (validation set)")
ax.set_xticks(SIZES); ax.set_ylim(0.4, 1.0)
ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6)
fig.savefig(f"{OUT}/FigS4_sigma_rank.png", dpi=600)
fig.savefig(f"{OUT}/FigS4_sigma_rank.pdf")
plt.close(fig)
print(f"→ {OUT}/FigS4_sigma_rank.png")

# ---------- 图 S5：分档 MAE ----------
pool = {"transfer": ([], []), "baseline": ([], [])}
for arm in ("transfer", "baseline"):
    for f in sorted(glob.glob(f"{VAL}/tr*_s*_{arm}.csv")):
        _, _, t, p = load(f)
        pool[arm][0].append(t); pool[arm][1].append(p)
ref = np.concatenate(pool["transfer"][0])
edges = np.quantile(ref, [0, .25, .5, .75, 1.0]); edges[0] -= 1e-6; edges[-1] += 1e-6
labels = [f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)]
fig, ax = plt.subplots(figsize=(6.4, 4.2))
wd = 0.36
for k, (arm, col) in enumerate((("transfer", C_T), ("baseline", C_B))):
    maes = []
    for i in range(len(edges) - 1):
        t = np.concatenate(pool[arm][0]); p = np.concatenate(pool[arm][1])
        m = (t >= edges[i]) & (t < edges[i + 1])
        maes.append(np.abs(p[m] - t[m]).mean())
    xs = np.arange(len(maes))
    ax.bar(xs + (k - .5) * wd, maes, wd, color=col, edgecolor="k", lw=.6,
           label="Transfer learning" if k == 0 else "Randomly initialized baseline")
    for x, v in zip(xs + (k - .5) * wd, maes):
        ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    print(f"  {arm} 分档 MAE:", [f"{v:.2f}" for v in maes])
ax.set_xticks(np.arange(len(labels))); ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel(r"Reference $\sigma_{\mathrm{opt}}$ bin (10$^{-16}$ cm$^2$)")
ax.set_ylabel(r"MAE of $\sigma_{\mathrm{opt}}$ (10$^{-16}$ cm$^2$)")
ax.legend(fontsize=9); ax.grid(alpha=.25, lw=.6, axis="y")
fig.savefig(f"{OUT}/FigS5_sigma_binned_mae.png", dpi=600)
fig.savefig(f"{OUT}/FigS5_sigma_binned_mae.pdf")
plt.close(fig)
print(f"→ {OUT}/FigS5_sigma_binned_mae.png")
print("分档边界:", [f"{e:.2f}" for e in edges], "| 池化样本数:", len(ref))
