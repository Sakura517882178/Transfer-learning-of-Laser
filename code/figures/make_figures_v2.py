#!/usr/bin/env python3
"""绿光激光论文：正文图 3 与 SI 图 S1–S5（全部基于 10 组划分 × 5 规模 × 2 臂的真实结果）
输出: /mnt/e/paper-lasergreen/version1/figures/
"""
import csv, glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

OUT = "/mnt/e/paper-lasergreen/version1/figures"
os.makedirs(OUT, exist_ok=True)
RUNS = "/mnt/f/green-laser/3core/runs/repeat2"
REP = "/mnt/f/green-laser/3core/report"
P = "/mnt/i/green-laser-3core"
SIZES = [6, 13, 27, 40, 53]
C_TR, C_BL = "#2E5C8A", "#B4552D"          # 迁移=蓝，基线=砖红（雅、非淡）
plt.rcParams.update({
    "font.size": 11, "font.family": "DejaVu Sans", "axes.linewidth": 1.2,
    "xtick.direction": "out", "ytick.direction": "out", "xtick.major.width": 1.1,
    "ytick.major.width": 1.1, "axes.labelsize": 12, "legend.frameon": False,
    "savefig.bbox": "tight", "figure.dpi": 120,
})


def style(ax):
    for s in ax.spines.values():
        s.set_linewidth(1.2)
        s.set_visible(True)
    ax.tick_params(top=False, right=False)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}", dpi=600)
    plt.close(fig)
    print("  →", f"{OUT}/{name}.png")


# ───────── 数据 ─────────
per = json.load(open(f"{REP}/repeat2_per_run.json"))
def arr(target, arm):
    m, s = [], []
    for n in SIZES:
        v = [per[f"{n}|{k}|{arm}"][target] for k in range(10) if f"{n}|{k}|{arm}" in per]
        m.append(np.mean(v)); s.append(np.std(v, ddof=1))
    return np.array(m), np.array(s)

# test7 逐样本预测
pred = {}                      # (n,arm) -> {name: (S1, sig)}
for f in sorted(glob.glob(f"{REP}/test7b/tr*.csv")):
    sid, arm = os.path.basename(f)[:-4].rsplit("_", 1)
    n = int(sid.split("_")[0][2:])
    for r in csv.DictReader(open(f)):
        nm = r["name"].replace("_valid", "").replace("_train", "")
        pred.setdefault((n, arm), {}).setdefault(nm, []).append(
            (float(r["target_0_pred"]), float(r["target_1_pred"])))
labels = {r["name"]: r for r in csv.DictReader(open(f"{P}/computed/labels_69.csv"))}
EXP = ["Th_Th_Fl", "Py_dTT_Fl"]

# 逐 epoch 曲线
EP, M0, M1, AG = (re.compile(r"epoch (\d+) train loss"), re.compile(r"target_0\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)"),
                  re.compile(r"target_1\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)"),
                  re.compile(r"AGGREGATE\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)"))


def parse(log):
    rows, cur = [], {}
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        if EP.search(line):
            cur = {"epoch": int(EP.search(line).group(1))}
        if M0.search(line) and cur:
            cur["s1"] = float(M0.search(line).group(2))
        if M1.search(line) and cur:
            cur["sg"] = float(M1.search(line).group(2))
        if AG.search(line) and cur:
            cur["tot"] = float(AG.search(line).group(2)); rows.append(cur); cur = {}
    return rows


# ───────── 图 3（正文）：(a) t-SNE  (b) S1 R² vs n  (c) σ R² vs n  (d) 配对 Δ + 95%CI ─────────
fig, axs = plt.subplots(1, 4, figsize=(20.5, 4.5))
plt.subplots_adjust(wspace=0.32)

# (a) t-SNE：源域 1010 vs 目标域 69（第 10 层 CLS）
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
zs, z0, z1 = (np.load(f"{REP}/align/{f}") for f in ("source_0.npz", "target_0.npz", "target_1.npz"))
Xs, Xt = zs["L10"].astype(float), np.vstack([z0["L10"], z1["L10"]]).astype(float)
tn = [n.replace("_valid", "").replace("_train", "") for n in list(z0["names"]) + list(z1["names"])]
X = np.vstack([Xs, Xt])
Z = TSNE(n_components=2, perplexity=30, random_state=0, init="pca").fit_transform(PCA(50).fit_transform(X))
ax = axs[0]
ax.scatter(Z[:len(Xs), 0], Z[:len(Xs), 1], s=6, c="#9DB7CE", alpha=.55, lw=0, label=f"Source (n={len(Xs)})")
ax.scatter(Z[len(Xs):, 0], Z[len(Xs):, 1], s=28, c="#C0504D", alpha=.9, lw=.4, ec="w", label=f"Target (n={len(Xt)})")
for i, nm in enumerate(tn):
    if nm in EXP:
        ax.annotate(nm, (Z[len(Xs) + i, 0], Z[len(Xs) + i, 1]), fontsize=9, xytext=(-6, -12),
                    textcoords="offset points", color="#7F1D1D", ha="right")
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); style(ax)
ax.set_title("(a) Representation space (Uni-Mol encoder, layer 10)", fontsize=11)
ax.legend(loc="best", fontsize=9)

# (b)(c) 性能 vs 规模
for k, (tgt, lab, ttl) in enumerate((("s1_r2", "S$_1$ energy, R$^2$", "(b) S$_1$ energy prediction"),
                                     ("sg_r2", r"$\sigma_{\mathrm{opt}}$, R$^2$", r"(c) Net emission cross-section"))):
    ax = axs[1 + k]
    for arm, c, mk, nm in (("transfer", C_TR, "o", "Transfer learning"), ("baseline", C_BL, "s", "Random-init baseline")):
        m, s = arr(tgt, arm)
        ax.errorbar(SIZES, m, yerr=s, color=c, marker=mk, ms=6, lw=1.8, capsize=3.5, label=nm)
    ax.axhline(0, color="gray", lw=.8, ls=":")
    ax.set_xlabel("Number of training molecules"); ax.set_ylabel(lab)
    ax.set_xticks(SIZES); ax.set_title(ttl, fontsize=11); style(ax)
    if k == 0:
        ax.legend(loc="lower right", fontsize=9)

# (d) 配对 Δ + 95%CI
ax = axs[3]
T95 = {4: 2.776, 9: 2.262}
for j, (tgt, c, off) in enumerate((("tot_r2", "#2E5C8A", -0.16), ("sg_r2", "#7E9FBF", 0.0), ("s1_r2", "#C98A5E", 0.16))):
    mu, ci = [], []
    for n in SIZES:
        d = np.array([per[f"{n}|{k}|transfer"][tgt] - per[f"{n}|{k}|baseline"][tgt]
                      for k in range(10) if f"{n}|{k}|transfer" in per])
        mu.append(d.mean()); ci.append(T95.get(len(d) - 1, 2.262) * d.std(ddof=1) / np.sqrt(len(d)))
    ax.errorbar(np.array(SIZES) + off, mu, yerr=ci, fmt="o", ms=5, lw=1.6, capsize=3,
                color=c, label={"tot_r2": "Total R$^2$", "sg_r2": r"$\sigma_{\mathrm{opt}}$ R$^2$", "s1_r2": "S$_1$ R$^2$"}[tgt])
ax.axhline(0, color="k", lw=1)
ax.set_xlabel("Number of training molecules"); ax.set_ylabel(r"Paired $\Delta$ (transfer $-$ baseline)")
ax.set_xticks(SIZES); ax.set_title("(d) Paired difference, mean ± 95% CI", fontsize=11)
ax.set_ylim(-0.42, 0.50)
style(ax); ax.legend(loc="lower left", fontsize=8.5, ncol=1)
save(fig, "Fig3_transfer_learning_v2")

# ───────── SI 图 S1：性能 vs 规模（S1 R²、σ R²、agg MAE 三面板）─────────
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.3))
for ax, (tgt, lab, ttl) in zip(axs, (("s1_r2", "S$_1$ energy, R$^2$", "(a) S$_1$ energy"),
                                     ("sg_r2", r"$\sigma_{\mathrm{opt}}$, R$^2$", r"(b) $\sigma_{\mathrm{opt}}$"),
                                     ("agg", "Aggregate MAE", "(c) Aggregate MAE"))):
    for arm, c, mk, nm in (("transfer", C_TR, "o", "Transfer learning"), ("baseline", C_BL, "s", "Random-init baseline")):
        m, s = arr(tgt, arm)
        ax.errorbar(SIZES, m, yerr=s, color=c, marker=mk, ms=5.5, lw=1.8, capsize=3, label=nm)
    ax.set_xlabel("Number of training molecules"); ax.set_ylabel(lab); ax.set_xticks(SIZES)
    ax.set_title(ttl, fontsize=11); style(ax)
    if tgt == "s1_r2":
        ax.legend(loc="lower right", fontsize=9)
fig.suptitle("Model performance versus training size (mean ± s.d. over 10 independent splits, validation set)",
             fontsize=11.5, y=1.0)
save(fig, "FigS1_performance_vs_size_v2")

# ───────── SI 图 S2：锁定测试集逐样本预测（7 分子，n=53）─────────
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
names7 = sorted({nm for (n, a) in pred if n == 53 for nm in pred[(n, a)]})
x = np.arange(len(names7))
for ax, (j, lab, ttl, unit) in zip(axs, ((0, "S$_1$ energy (eV)", "(a) S$_1$ energy on locked test set", "eV"),
                                         (1, r"$\sigma_{\mathrm{opt}}$ (10$^{-16}$ cm$^2$)",
                                          r"(b) $\sigma_{\mathrm{opt}}$ on locked test set", ""))):
    ref = np.array([float(labels[nm]["S1_eV"] if j == 0 else labels[nm]["Sigma_opt"]) for nm in names7])
    ax.bar(x - 0.2, ref, 0.4, color="#B9B9B9", label="DFT reference")
    for k, (arm, c) in enumerate((("transfer", C_TR), ("baseline", C_BL))):
        vals = [np.mean([p[j] for p in pred[(53, arm)][nm]]) for nm in names7]
        ax.bar(x + 0.05 + k * 0.2, vals, 0.18, color=c,
               label="Transfer learning" if k == 0 else "Random-init baseline")
    ax.set_xticks(x); ax.set_xticklabels(names7, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel(lab); ax.set_title(ttl, fontsize=11); style(ax)
    if j == 0:
        ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
save(fig, "FigS2_locked_test_predictions_v2")

# ───────── SI 图 S3：逐 epoch 验证曲线（n=13 与 n=53）─────────
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.4))
for ax, n in zip(axs, (13, 53)):
    for arm, c, nm in (("transfer", C_TR, "Transfer learning"), ("baseline", C_BL, "Random-init baseline")):
        lg = f"{RUNS}/tr{n}_s00/{arm}/save_finetune/train.log"
        if not os.path.exists(lg):
            continue
        rows = [r for r in parse(lg) if "tot" in r]
        e = [r["epoch"] for r in rows]; v = [r["tot"] for r in rows]
        ax.plot(e, v, color=c, lw=1.3, label=f"{nm} (s.d. {np.std(v, ddof=1):.2f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation total R$^2$")
    ax.set_title(f"(n = {n})", fontsize=11); style(ax); ax.legend(fontsize=9, loc="lower right")
fig.suptitle("Epoch-resolved validation metrics (split 0) — fluctuation between adjacent epochs", fontsize=11.5, y=1.0)
save(fig, "FigS3_epoch_curves_v2")

# ───────── SI 图 S4/S5：σopt 排序相关 + 分档 MAE ─────────
def sigma_points():
    pr, rf = [], []
    for (n, arm), d in pred.items():
        for nm, vals in d.items():
            pr.append(np.mean([v[1] for v in vals])); rf.append(float(labels[nm]["Sigma_opt"]))
    return np.array(pr), np.array(rf)


pr, rf = sigma_points()
from scipy.stats import spearmanr, pearsonr
rho = spearmanr(rf, pr).correlation
r2 = pearsonr(rf, pr)[0] ** 2
fig, ax = plt.subplots(figsize=(5.6, 5))
ax.scatter(rf, pr, s=16, c="#2E5C8A", alpha=.55, lw=0)
lim = [min(rf.min(), pr.min()) - .3, max(rf.max(), pr.max()) + .3]
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(r"DFT reference $\sigma_{\mathrm{opt}}$ (10$^{-16}$ cm$^2$)")
ax.set_ylabel(r"Predicted $\sigma_{\mathrm{opt}}$ (10$^{-16}$ cm$^2$)")
ax.text(.05, .93, f"Spearman $\\rho$ = {rho:.2f}\n$R^2$ = {r2:.2f}", transform=ax.transAxes, va="top", fontsize=11)
style(ax); ax.legend(loc="lower right", fontsize=9)
save(fig, "FigS4_sigma_rank_v2")

bins = [(-2, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
mb, e_ = [], []
for lo, hi in bins:
    m = (rf >= lo) & (rf < hi)
    e_.append(np.abs(pr[m] - rf[m]).mean() if m.sum() else np.nan)
    mb.append(f"{lo}–{hi}")
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.bar(range(len(bins)), e_, color="#7E9FBF", width=.62, edgecolor="k", lw=.8)
ax.set_xticks(range(len(bins))); ax.set_xticklabels(mb, fontsize=10)
ax.set_xlabel(r"Reference $\sigma_{\mathrm{opt}}$ bin (10$^{-16}$ cm$^2$)")
ax.set_ylabel("Mean absolute error")
ax.set_title(r"Binned MAE of $\sigma_{\mathrm{opt}}$ (all splits and sizes)", fontsize=11)
style(ax)
save(fig, "FigS5_sigma_binned_mae_v2")
print("全部完成 →", OUT)
