#!/usr/bin/env python3
"""σ 修正（×π）后的下游重算：
 ① test7 独立评估（预测值 ×π，R²/ρ 不变，MAE ×π）② 主批次 valid 指标（sg_mae×π、agg 重算）
 ③ 330 全库预测与 FOM（×π）④ 两个实验分子的最终预测
输出: report/{test7_stats_new.json, repeat5_metrics_new.json, predict330_all_new.csv, predict330_FOM_new.csv}
"""
import csv, glob, json, os
import numpy as np
from scipy.stats import spearmanr, pearsonr

REP = "/mnt/f/green-laser/3core/report"
P = "/mnt/i/green-laser-3core"
PI = np.pi
lab = {r["name"]: r for r in csv.DictReader(open(f"{P}/computed/labels_69.csv"))}
HC = 1239.84193

# ① test7
s7 = {}
for f in sorted(glob.glob(f"{REP}/test7/*.csv")):
    sid, arm = os.path.basename(f)[:-4].rsplit("_", 1)
    n = int(sid.split("_")[0][2:])
    for r in csv.DictReader(open(f)):
        nm = r["name"].replace("_valid", "").replace("_train", "")
        s7.setdefault((n, arm), {}).setdefault(nm, []).append(
            (float(r["target_0_pred"]), float(r["target_1_pred"]) * PI))   # σ ×π
out7 = {}
for (n, arm), d in s7.items():
    s1p = np.array([np.mean([v[0] for v in vs]) for vs in d.values()])
    sgp = np.array([np.mean([v[1] for v in vs]) for vs in d.values()])
    names = list(d)
    s1t = np.array([float(lab[x]["S1_eV"]) for x in names])
    sgt = np.array([float(lab[x]["Sigma_opt"]) for x in names])
    out7[f"{n}|{arm}"] = {
        "S1": {"mae": float(np.abs(s1p - s1t).mean()), "rho": float(spearmanr(s1t, s1p).correlation),
               "r2": float(pearsonr(s1t, s1p)[0] ** 2)},
        "sigma": {"mae": float(np.abs(sgp - sgt).mean()), "rho": float(spearmanr(sgt, sgp).correlation),
                  "r2": float(pearsonr(sgt, sgp)[0] ** 2)},
        "pred": {x: [float(a), float(b)] for x, (a, b) in zip(names, zip(s1p, sgp))},
        "true": {x: [float(lab[x]["S1_eV"]), float(lab[x]["Sigma_opt"])] for x in names}}
json.dump(out7, open(f"{REP}/test7_stats_new.json", "w"), indent=1)
print("① test7 重算完成")
for n in (6, 13, 27, 40, 53):
    for arm in ("transfer", "baseline"):
        k = f"{n}|{arm}"
        if k in out7:
            print(f"   n={n:<3}{arm:<10} S1 MAE {out7[k]['S1']['mae']:.3f} eV ρ{out7[k]['S1']['rho']:.2f} | "
                  f"σ MAE {out7[k]['sigma']['mae']:.3f} ρ{out7[k]['sigma']['rho']:.2f}")

# ② 主批次 valid 指标
per = json.load(open(f"{REP}/repeat5_per_run.json"))
new = {}
for k, v in per.items():
    w = dict(v)
    w["sg_mae"] = v["sg_mae"] * PI                       # σ MAE ×π
    w["agg_scaled"] = (v["s1_mae"] + v["sg_mae"] * PI) / 2
    new[k] = w
json.dump(new, open(f"{REP}/repeat5_metrics_new.json", "w"), indent=1)
print("\n② 主批次指标已按 ×π 重算（R² 不变）")
SIZES = [6, 13, 27, 40, 53]
print(f"   {'n':>3} {'agg(迁移)':>12} {'agg(基线)':>12} {'σMAE(迁移)':>12} {'R²σ(迁移)':>10}")
for n in SIZES:
    t = [new[f"{n}|{s}|transfer"] for s in range(10)]
    b = [new[f"{n}|{s}|baseline"] for s in range(10)]
    print(f"   {n:>3} {np.mean([x['agg_scaled'] for x in t]):>12.3f} {np.mean([x['agg_scaled'] for x in b]):>12.3f} "
          f"{np.mean([x['sg_mae'] for x in t]):>12.3f} {np.mean([x['sg_r2'] for x in t]):>10.3f}")

# ③ 330 全库
sa = {r["name"]: float(r["SA"]) for r in csv.DictReader(open(f"{P}/library/library_330_SA.csv")) if r["SA"]}
pr = {r["name"]: (float(r["S1_eV_pred"]), float(r["Sigma_opt_pred"]) * PI)
      for r in csv.DictReader(open(f"{REP}/predict330_all.csv"))}
rows = [{"name": n, "s1": v[0], "wl": HC / v[0] if v[0] > 0 else None, "sg": v[1],
         "sa": sa.get(n), "fom": (v[1] / sa[n] if sa.get(n) else None)} for n, v in pr.items()]
green = sorted([r for r in rows if r["wl"] and r["wl"] > 500 and r["fom"]], key=lambda r: -r["fom"])
rank = {r["name"]: i for i, r in enumerate(green, 1)}
with open(f"{REP}/predict330_FOM_new.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["name", "S1_eV_pred", "lambda_nm_pred", "Sigma_opt_pred", "SA", "FOM_sigma_SA", "rank_FOM_green"])
    for r in sorted([x for x in rows if x["fom"]], key=lambda x: -x["fom"]):
        w.writerow([r["name"], f"{r['s1']:.4f}", f"{r['wl']:.1f}", f"{r['sg']:.4f}", f"{r['sa']:.3f}",
                    f"{r['fom']:.4f}", rank.get(r["name"], "")])
print("\n③ 330 全库 σ 与 FOM 已 ×π → predict330_FOM_new.csv")
print("   绿光区 FOM 前 5:")
for i, r in enumerate(green[:5], 1):
    print(f"     {i} {r['name']:<18}{r['wl']:>6.0f} nm σopt={r['sg']:>7.3f} SA={r['sa']:.2f} FOM={r['fom']:.3f}")

# ④ 两个实验分子
print("\n④ 两个实验分子（n=53，10 组划分均值）:")
for tgt in ("Th_Th_Fl", "Py_dTT_Fl"):
    tr = [out7[f"53|transfer"]["pred"][tgt] for _ in range(1)][0]
    preds = [out7[f"53|transfer"]["pred"][tgt], out7[f"53|baseline"]["pred"][tgt]]
    t = out7["53|transfer"]["true"][tgt]
    print(f"   {tgt:<12} 参考 S1={t[0]:.3f} eV ({HC/t[0]:.0f} nm) σopt={t[1]:.3f} | "
          f"迁移预测 S1={preds[0][0]:.3f} ({HC/preds[0][0]:.0f} nm) σopt={preds[0][1]:.3f} | "
          f"基线 σopt={preds[1][1]:.3f}")
