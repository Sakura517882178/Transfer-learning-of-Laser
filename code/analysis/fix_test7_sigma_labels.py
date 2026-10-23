#!/usr/bin/env python3
"""修正公开仓库 results/test7/ 的 σopt 真值列（前置：2026-09-19 的 σ 缺 4π 修正）。

背景：results/test7/*.csv 是 2026-09-19 那次 test7b 评估的逐 run 预测，但其
`target_1_true` 仍是"修 π 之前"的旧标签（= 修正值 / π），而 `target_1_pred` 来自
已用修正后标签训练的模型（尺度正确）⇒ 两者差 π，σ 的 MAE/R² 全是假的（R²≈−50~−82）。
S₁ 列（target_0_*）不受影响；模型训练与预测本身没有问题。

本脚本做的事（**不重训、不动预测值**，幂等）：
  1. 用 labels_69.csv 的现行 σopt 覆盖 `target_1_true`；
  2. 逐行分类：待修（旧×π 吻合）/ 已修正 / 两者都对不上（异常，exit 1）；
  3. 重算 S₁/σ 的 R²/ρ/MAE → results/summary/test7_{summary,per_run}.json。

用法：
  /home/sakura/app/anaconda3/bin/python3 code/analysis/fix_test7_sigma_labels.py
  加 --dry-run 只报数不写盘；--test7/--labels/--summary 可指向临时目录做自测。
"""
import argparse, csv, glob, json, math, os, sys

REPO = "/mnt/e/paper-lasergreen/version2/github-repo"
ap = argparse.ArgumentParser()
ap.add_argument("--test7", default=f"{REPO}/results/test7", help="逐 run CSV 目录")
ap.add_argument("--labels", default="/mnt/i/green-laser-3core/computed/labels_69.csv")
ap.add_argument("--summary", default=f"{REPO}/results/summary", help="指标输出目录")
ap.add_argument("--dry-run", action="store_true")
opt = ap.parse_args()
APPLY = not opt.dry_run
HC = 1239.84193  # eV·nm（S₁ → 波长，仅打印用）

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def strip(n):
    return n.replace("_valid", "").replace("_train", "")


labels = {r["name"]: float(r["Sigma_opt"]) for r in csv.DictReader(open(opt.labels))}
files = sorted(glob.glob(f"{opt.test7}/tr*.csv"))
assert files, f"没有找到 {opt.test7}/tr*.csv"

# ---- 1) 校验 + 覆写真值列（幂等：已修正的文件再跑只报"已修正"） ----
n_rows = n_prefix = n_already = n_bad = 0
worst = 0.0
for f in files:
    rows = list(csv.DictReader(open(f)))
    hdr = list(rows[0].keys())
    for r in rows:
        nm = strip(r["name"])
        old, new = float(r["target_1_true"]), labels[nm]
        d_prefix, d_same = abs(old * math.pi - new), abs(old - new)
        n_rows += 1
        if d_same < 1e-4:
            n_already += 1
        elif d_prefix < 1e-4:
            n_prefix += 1
            worst = max(worst, d_prefix)
        else:
            n_bad += 1
            print(f"  ! 两者都对不上 {os.path.basename(f)} {nm}: old={old} old*pi={old*math.pi} new={new}")
        r["target_1_true"] = f"{new:.6f}"
    if APPLY:
        with open(f, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=hdr, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)

print(f"[真值列] {len(files)} 个 CSV / {n_rows} 行：待修（旧×π 吻合）{n_prefix}，已修正 {n_already}，"
      f"异常 {n_bad}，最大偏差 {worst:.2e}{'' if APPLY else '（dry-run 未写）'}")
if n_bad:
    sys.exit(1)

# ---- 2) 重算指标 ----
rec = {}
REF = {"Th_Th_Fl": (2.296011, 4.009933, 540.0), "Py_dTT_Fl": (2.100988, 4.838591, 590.1)}
for f in files:
    sid, arm = os.path.basename(f)[:-4].rsplit("_", 1)
    n, s = int(sid.split("_")[0][2:]), int(sid.split("_")[1][1:])
    rows = list(csv.DictReader(open(f)))
    names = [strip(r["name"]) for r in rows]
    y = np.array([[float(r["target_0_true"]), float(r["target_1_true"])] for r in rows])
    p = np.array([[float(r["target_0_pred"]), float(r["target_1_pred"])] for r in rows])
    d = {}
    for j, t in enumerate(("S1_eV", "Sigma_opt")):
        d[t] = dict(r2=float(r2_score(y[:, j], p[:, j])),
                    rho=float(spearmanr(y[:, j], p[:, j]).correlation),
                    mae=float(np.mean(np.abs(y[:, j] - p[:, j]))))
    d["pred"] = {nm: (float(pp[0]), float(pp[1])) for nm, pp in zip(names, p)}
    d["true"] = {nm: (float(yy[0]), float(yy[1])) for nm, yy in zip(names, y)}
    rec[(n, s, arm)] = d

sizes = sorted({k[0] for k in rec})
pre = {"S1_eV": "s1", "Sigma_opt": "sg"}  # JSON 字段前缀
agg = {}
for n in sizes:
    for arm in ("baseline", "transfer"):
        rs = [rec[(n, s, arm)] for s in range(10) if (n, s, arm) in rec]
        if not rs:
            continue
        agg[(n, arm)] = {f"{pre[t]}_{m}": float(np.mean([r[t][m] for r in rs]))
                         for t in ("S1_eV", "Sigma_opt") for m in ("r2", "rho", "mae")}
        agg[(n, arm)]["k"] = len(rs)

print(f"\n{'n':>3} | {'基线 S1 R²/ρ/MAE | σ R²/ρ/MAE':>44} | {'迁移 S1 R²/ρ/MAE | σ R²/ρ/MAE':>44}")
for n in sizes:
    def fmt(x):
        return (f"{x['s1_r2']:+.3f}/{x['s1_rho']:+.2f}/{x['s1_mae']:.3f} | "
                f"{x['sg_r2']:+.3f}/{x['sg_rho']:+.2f}/{x['sg_mae']:.3f}") if x else "—"
    print(f"{n:>3} | {fmt(agg.get((n, 'baseline'))):>44} | {fmt(agg.get((n, 'transfer'))):>44}")

print("\n实验分子（各规模各臂均值，eV / nm / σ）:")
for nm, (r_s1, r_sg, r_wl) in REF.items():
    print(f"  {nm}: 参考 S1={r_s1:.3f} eV ({r_wl:.0f} nm), σ={r_sg:.3f}")
    for n in sizes:
        cells = []
        for arm in ("baseline", "transfer"):
            pr = [rec[(n, s, arm)]["pred"][nm] for s in range(10)
                  if (n, s, arm) in rec and nm in rec[(n, s, arm)]["pred"]]
            if pr:
                m = np.array(pr).mean(axis=0)
                cells.append(f"{arm[:4]}: {m[0]:.3f} eV ({HC / m[0]:.0f} nm) σ={m[1]:.3f}")
        print(f"     n={n:<3} " + " | ".join(cells))

if APPLY:
    os.makedirs(opt.summary, exist_ok=True)
    json.dump({f"{k[0]}|{k[1]}": v for k, v in agg.items()},
              open(f"{opt.summary}/test7_summary.json", "w"), indent=2, ensure_ascii=False)
    json.dump({f"{k[0]}|{k[1]}|{k[2]}": {t: v[t] for t in ("S1_eV", "Sigma_opt")} for k, v in rec.items()},
              open(f"{opt.summary}/test7_per_run.json", "w"), indent=2, ensure_ascii=False)
    print(f"\n→ {opt.summary}/test7_summary.json + test7_per_run.json")
