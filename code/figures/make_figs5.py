#!/usr/bin/env python3
"""SI 图 S5 重做：σopt 分档平均绝对误差（按参考值分位分档，自适应新 σ 尺度）
数据：330 全库预测（p330new_all.csv）× 69 个 DFT 参考值（labels_69.csv）
输出：E:\paper-lasergreen\version1\figures\FigS5_sigma_binned_mae_v2.{png,pdf}
"""
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REP = "/mnt/f/green-laser/3core/report"
P = "/mnt/i/green-laser-3core"
OUT = "/mnt/e/paper-lasergreen/version1/figures"
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "xtick.direction": "out", "ytick.direction": "out", "savefig.bbox": "tight",
                     "figure.dpi": 130})

pred = {r["name"]: float(r["Sigma_opt_pred"]) for r in csv.DictReader(open(f"{REP}/p330new_all.csv"))
        if r["Sigma_opt_pred"]}
lab = {r["name"]: float(r["Sigma_opt"]) for r in csv.DictReader(open(f"{P}/computed/labels_69.csv"))}


def conv(n):
    p = n.split("_")
    if p[-2:] == ["Fl", "Ph"]:
        t, b = "PhFl", (p[1] if len(p) >= 3 else "")
    else:
        t, b = p[-1], (p[1] if len(p) >= 3 else "")
    b = "Ph" if b == "P" else b
    return f"{p[0]}_{t}" if not b else f"{p[0]}_{b}_{t}"


pairs = [(lab[k], pred[conv(k)]) for k in lab if conv(k) in pred]
ref = np.array([a for a, _ in pairs]); prd = np.array([b for _, b in pairs])
edges = np.quantile(ref, [0, .25, .5, .75, 1.0])
edges[0] -= 1e-6; edges[-1] += 1e-6
print(f"有效样本 {len(ref)} 个 | 参考 σopt 范围 {ref.min():.2f}–{ref.max():.2f}")
print(f"分档边界（四分位）: {np.round(edges,2)}")

fig, ax = plt.subplots(figsize=(6.4, 4.4))
w = 0.36
xs = []
for i in range(len(edges) - 1):
    m = (ref >= edges[i]) & (ref < edges[i + 1])
    if m.sum() == 0:
        continue
    xs.append(i)
    mae = np.abs(prd[m] - ref[m]).mean()
    bias = (prd[m] - ref[m]).mean()
    x = len(xs) - 1
    ax.bar(x - w / 2, mae, w, color="#1F4E79", edgecolor="k", lw=.6, label="MAE" if x == 0 else None)
    ax.bar(x + w / 2, bias, w, color="#4E9F9B", edgecolor="k", lw=.6,
           label="Mean signed error" if x == 0 else None)
    ax.text(x - w / 2, mae, f"{mae:.2f}", ha="center", va="bottom", fontsize=9)
    ax.text(x + w / 2, bias, f"{bias:+.2f}", ha="center",
            va="bottom" if bias >= 0 else "top", fontsize=9)
    print(f"  档 {i+1} [{edges[i]:.2f}, {edges[i+1]:.2f}): n={int(m.sum()):2d} | MAE {mae:.3f} | 偏差 {bias:+.3f}")
ax.axhline(0, color="k", lw=.9)
ax.set_xticks(range(len(xs)))
ax.set_xticklabels([f"{edges[i]:.1f}–{edges[i+1]:.1f}" for i in range(len(edges) - 1)], fontsize=9)
ax.set_xlabel(r"Reference $\sigma_{\mathrm{opt}}$ bin (10$^{-16}$ cm$^2$)")
ax.set_ylabel(r"Error in $\sigma_{\mathrm{opt}}$ (10$^{-16}$ cm$^2$)")
ax.set_title(r"Binned error of $\sigma_{\mathrm{opt}}$ (69 DFT references)", fontsize=11)
ax.legend(fontsize=9)
for sp in ax.spines.values():
    sp.set_linewidth(1.1)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/FigS5_sigma_binned_mae_v2.{ext}", dpi=600)
plt.close(fig)
print(f"→ {OUT}/FigS5_sigma_binned_mae_v2.png")
