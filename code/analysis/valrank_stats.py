#!/usr/bin/env python3
"""验证集排序能力统计：逐 run 计算 Spearman ρ（S₁、σopt），按规模×臂聚合
输出：report/valrank_stats.json + 屏幕表格（10 组划分均值±SD）
"""
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import spearmanr

W = "/mnt/f/green-laser/3core"
OUT = f"{W}/report/valrank"
SIZES = [6, 13, 27, 40, 53]


def load(f):
    rows = list(csv.DictReader(open(f)))
    s1t = np.array([float(r["target_0_true"]) for r in rows])
    s1p = np.array([float(r["target_0_pred"]) for r in rows])
    sgt = np.array([float(r["target_1_true"]) for r in rows])
    sgp = np.array([float(r["target_1_pred"]) for r in rows])
    return spearmanr(s1t, s1p).correlation, spearmanr(sgt, sgp).correlation, len(rows)


per = {}
for f in sorted(glob.glob(f"{OUT}/tr*_s*_*.csv")):
    base = os.path.basename(f)[:-4]              # tr13_s00_transfer
    sid, arm = base.rsplit("_", 1)
    n = int(sid.split("_")[0][2:])
    seed = int(sid.split("_s")[1])
    r1, r2, k = load(f)
    per[f"{n}|{seed}|{arm}"] = {"s1_rho": r1, "sg_rho": r2, "n_val": k}
json.dump(per, open(f"{W}/report/valrank_stats.json", "w"), indent=1, ensure_ascii=False)
print(f"解析 run: {len(per)}/100\n")

print(f"{'n':>3}{'臂':<10}{'S₁ ρ':>16}{'σopt ρ':>16}{'验证集分子数':>12}")
summary = {}
for n in SIZES:
    for arm in ("baseline", "transfer"):
        v1 = [per[f"{n}|{s}|{arm}"]["s1_rho"] for s in range(10) if f"{n}|{s}|{arm}" in per]
        v2 = [per[f"{n}|{s}|{arm}"]["sg_rho"] for s in range(10) if f"{n}|{s}|{arm}" in per]
        kv = [per[f"{n}|{s}|{arm}"]["n_val"] for s in range(10) if f"{n}|{s}|{arm}" in per]
        if not v1:
            continue
        summary[f"{n}|{arm}"] = {"s1_rho": float(np.mean(v1)), "s1_sd": float(np.std(v1, ddof=1)),
                                 "sg_rho": float(np.mean(v2)), "sg_sd": float(np.std(v2, ddof=1)),
                                 "k": len(v1), "n_val": int(np.mean(kv))}
        print(f"{n:>3} {arm:<10}{np.mean(v1):>10.3f}±{np.std(v1,ddof=1):<5.3f}{np.mean(v2):>10.3f}±{np.std(v2,ddof=1):<5.3f}{int(np.mean(kv)):>12}")
json.dump(summary, open(f"{W}/report/valrank_summary.json", "w"), indent=1, ensure_ascii=False)
print(f"\n配对 Δρ（迁移 − 基线，逐划分）:")
for n in SIZES:
    d1 = [per[f"{n}|{s}|transfer"]["s1_rho"] - per[f"{n}|{s}|baseline"]["s1_rho"] for s in range(10)
          if f"{n}|{s}|transfer" in per and f"{n}|{s}|baseline" in per]
    d2 = [per[f"{n}|{s}|transfer"]["sg_rho"] - per[f"{n}|{s}|baseline"]["sg_rho"] for s in range(10)
          if f"{n}|{s}|transfer" in per and f"{n}|{s}|baseline" in per]
    if d1:
        print(f"  n={n:<3} ΔS₁ ρ {np.mean(d1):+.3f}±{np.std(d1,ddof=1):.3f} ({sum(1 for x in d1 if x>0)}/{len(d1)} 正)"
              f" | Δσopt ρ {np.mean(d2):+.3f}±{np.std(d2,ddof=1):.3f} ({sum(1 for x in d2 if x>0)}/{len(d2)} 正)")
print(f"\n→ {W}/report/valrank_summary.json")
