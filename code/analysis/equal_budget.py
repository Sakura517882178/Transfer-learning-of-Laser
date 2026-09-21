#!/usr/bin/env python3
"""等训练预算口径（10/20/30 epoch）的统计与出图
数据: F:\green-laser\3core\runs\repeat2\tr*_s*\*   (10 组划分 × 5 规模 × 2 臂)
输出: report/equal_budget_summary.json + E:\paper-lasergreen\version1\figures\Fig_EqualBudget.*
"""
import glob, json, os, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

W = "/mnt/f/green-laser/3core"
OUT = "/mnt/e/paper-lasergreen/version1/figures"
SIZES = [6, 13, 27, 40, 53]
EPOCHS = [10, 20, 30]
T95 = {9: 2.262}
C_TR, C_BL = "#1F4E79", "#B4552D"
plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "axes.linewidth": 1.1,
                     "xtick.direction": "out", "ytick.direction": "out", "legend.frameon": False,
                     "axes.labelsize": 9.5, "savefig.bbox": "tight", "figure.dpi": 130})

EP = re.compile(r"epoch (\d+) train loss")
AG = re.compile(r"AGGREGATE\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
M0 = re.compile(r"target_0\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
M1 = re.compile(r"target_1\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")


def load(path):
    """返回 {epoch: {tot, agg, s1, s1mae, sg, sgmae}}"""
    d, cur = {}, None
    for line in open(path, errors="ignore").read().replace("\r", "\n").split("\n"):
        m = EP.search(line)
        if m:
            cur = int(m.group(1)); d.setdefault(cur, {})
        a, b, c = AG.search(line), M0.search(line), M1.search(line)
        if cur is None:
            continue
        if a:
            d[cur]["agg"], d[cur]["tot"] = float(a.group(1)), float(a.group(2))
        if b:
            d[cur]["s1mae"], d[cur]["s1"] = float(b.group(1)), float(b.group(2))
        if c:
            d[cur]["sgmae"], d[cur]["sg"] = float(c.group(1)), float(c.group(2))
    return d


runs = {}
for n in SIZES:
    for s in range(10):
        for arm in ("transfer", "baseline"):
            p = f"{W}/runs/repeat2/tr{n}_s{s:02d}/{arm}/save_finetune/train.log"
            if os.path.exists(p):
                runs[f"{n}|{s}|{arm}"] = load(p)
print(f"载入 run: {len(runs)}")

# ── 统计：10/20/30 epoch 的两臂均值与配对 Δ ──
summary = {}
print(f"\n{'n':>3} {'epoch':>6}{'迁移 总R²':>18}{'基线 总R²':>18}{'Δ总R²':>16}{'Δ95%CI':>9}{'为正':>6}"
      f"{'ΔσR²':>10}{'ΔσMAE':>10}")
for E in EPOCHS:
    for n in SIZES:
        tr = [runs[f"{n}|{s}|transfer"].get(E, {}) for s in range(10)]
        bl = [runs[f"{n}|{s}|baseline"].get(E, {}) for s in range(10)]
        ok = [(t, b) for t, b in zip(tr, bl) if "tot" in t and "tot" in b]
        if not ok:
            continue
        tt = np.array([t["tot"] for t, _ in ok]); bb = np.array([b["tot"] for _, b in ok])
        ts = np.array([t["sg"] for t, _ in ok]); bs = np.array([b["sg"] for _, b in ok])
        tm = np.array([t["sgmae"] for t, _ in ok]); bm = np.array([b["sgmae"] for _, b in ok])
        dt = tt - bb
        ci = T95[9] * dt.std(ddof=1) / np.sqrt(len(dt))
        summary[f"{n}|{E}"] = {"n": n, "epoch": E, "k": len(ok),
                               "transfer_tot": float(tt.mean()), "baseline_tot": float(bb.mean()),
                               "transfer_tot_sd": float(tt.std(ddof=1)), "baseline_tot_sd": float(bb.std(ddof=1)),
                               "delta_tot": float(dt.mean()), "delta_tot_ci95": float(ci),
                               "positive_splits": int((dt > 0).sum()),
                               "delta_sg": float((ts - bs).mean()), "delta_sgmae": float((bm - tm).mean())}
        v = summary[f"{n}|{E}"]
        print(f"{n:>3} {E:>6}{v['transfer_tot']:>10.3f}±{v['transfer_tot_sd']:<7.3f}"
              f"{v['baseline_tot']:>10.3f}±{v['baseline_tot_sd']:<7.3f}"
              f"{v['delta_tot']:>+10.3f}±{v['delta_tot_ci95']:<6.3f}{v['positive_splits']:>4}/10"
              f"{v['delta_sg']:>+10.3f}{v['delta_sgmae']:>+10.3f}")
    print()
json.dump(summary, open(f"{W}/report/equal_budget_summary.json", "w"), indent=1, ensure_ascii=False)
print(f"→ {W}/report/equal_budget_summary.json")

# ── 出图 ──
fig = plt.figure(figsize=(19, 9.6))
gs = GridSpec(2, 5, figure=fig, hspace=0.42, wspace=0.34, height_ratios=[1.05, 1])

# 上排：5 个规模的总 R² vs epoch（两臂 mean±SD）
for i, n in enumerate(SIZES):
    ax = fig.add_subplot(gs[0, i])
    for arm, c, ls, nm in (("transfer", C_TR, "-", "Transfer"), ("baseline", C_BL, "--", "Random-init")):
        curves = []
        for s in range(10):
            d = runs.get(f"{n}|{s}|{arm}", {})
            ep = sorted(e for e in d if e <= 60 and "tot" in d[e])
            curves.append([d[e]["tot"] for e in ep])
        L = min(len(x) for x in curves) if curves else 0
        if not L:
            continue
        A = np.array([x[:L] for x in curves])
        m, sd = A.mean(0), A.std(0, ddof=1)
        ax.plot(range(1, L + 1), m, color=c, lw=1.6, ls=ls, label=nm)
        ax.fill_between(range(1, L + 1), m - sd, m + sd, color=c, alpha=.18, lw=0)
    for E in EPOCHS:
        ax.axvline(E, color="gray", lw=.6, ls=":")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Validation total R$^2$" if i == 0 else "")
    ax.set_title(f"n = {n}", fontsize=10)
    ax.set_xlim(1, 60); ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    if i == 0:
        ax.legend(fontsize=8.5, loc="lower right")
        ax.text(-0.30, 1.14, "(a)", transform=ax.transAxes, fontsize=13)

# 下排：Δ（迁移 − 基线）在 10/20/30 epoch 下的柱状 + CI
for j, (key, ylab, ttl) in enumerate((("delta_tot", r"$\Delta$ Total R$^2$", "(b) Total R$^2$: transfer $-$ baseline"),
                                      ("delta_sg", r"$\Delta$ $\sigma_{\mathrm{opt}}$ R$^2$",
                                       r"(c) $\sigma_{\mathrm{opt}}$ R$^2$: transfer $-$ baseline"))):
    ax = fig.add_subplot(gs[1, j * 2:j * 2 + 2])
    x = np.arange(len(SIZES)); w = 0.26
    for k, E in enumerate(EPOCHS):
        vals = [summary[f"{n}|{E}"][key] for n in SIZES]
        cis = [summary[f"{n}|{E}"]["delta_tot_ci95"] for n in SIZES]
        ax.bar(x + (k - 1) * w, vals, w, yerr=cis, capsize=2.5,
               color=["#1F4E79", "#5B8DB8", "#A8C4DA"][k], edgecolor="k", lw=.6, label=f"{E} epochs",
               error_kw=dict(lw=.8))
    ax.axhline(0, color="k", lw=1)
    ax.set_xticks(x); ax.set_xticklabels([f"n = {n}" for n in SIZES], fontsize=9)
    ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=10)
    for sp in ax.spines.values():
        sp.set_linewidth(1.1)
    if j == 0:
        ax.legend(fontsize=8.5, loc="upper right", ncol=3)
        ax.text(-0.10, 1.16, "(b)", transform=ax.transAxes, fontsize=13)
    else:
        ax.text(-0.10, 1.16, "(c)", transform=ax.transAxes, fontsize=13)
for ext in ("png", "pdf"):
    fig.savefig(f"{OUT}/Fig_EqualBudget.{ext}", dpi=600)
plt.close(fig)
print(f"→ {OUT}/Fig_EqualBudget.png / .pdf")
