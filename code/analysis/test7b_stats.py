#!/usr/bin/env python3
"""固定 test 7（7 个锁定分子，含 2 个实验分子）上的独立评估：
逐 run 计算 R²/ρ/MAE（S1、σ），按 规模×臂 聚合，并给出配对 Δ 与两个实验分子的预测值。"""
import csv, glob, os, json, re
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

OUT = "/mnt/f/green-laser/3core/report/test7b"
EXP = {"Th_Th_Fl": (2.296011, 1.276401, 540.0), "Py_dTT_Fl": (2.100988, 1.540171, 590.1)}
HC = 1239.84193  # eV·nm


def strip(n):
    return n.replace("_valid", "").replace("_train", "")


rec = {}
for f in sorted(glob.glob(f"{OUT}/tr*.csv")):
    sid, arm = os.path.basename(f)[:-4].rsplit("_", 1)
    n, s = int(sid.split("_")[0][2:]), int(sid.split("_")[1][1:])
    rows = list(csv.DictReader(open(f)))
    names = [strip(r["name"]) for r in rows]
    y = np.array([[float(r["target_0_true"]), float(r["target_1_true"])] for r in rows])
    p = np.array([[float(r["target_0_pred"]), float(r["target_1_pred"])] for r in rows])
    d = {}
    for j, t in enumerate(("S1_eV", "Sigma_opt")):
        d[t] = dict(r2=r2_score(y[:, j], p[:, j]), rho=spearmanr(y[:, j], p[:, j]).correlation,
                    mae=float(np.mean(np.abs(y[:, j] - p[:, j]))))
    d["pred"] = {nm: (float(pp[0]), float(pp[1])) for nm, pp in zip(names, p)}
    d["true"] = {nm: (float(yy[0]), float(yy[1])) for nm, yy in zip(names, y)}
    rec[(n, s, arm)] = d

sizes = sorted({k[0] for k in rec})
print(f"解析 {len(rec)} 个 run 的 test7 预测\n")
print(f"{'n':>3} | {'基线: S1 R²/ρ | σ R²/ρ | MAE':>40} | {'迁移: S1 R²/ρ | σ R²/ρ | MAE':>40}")
agg = {}
for n in sizes:
    for arm in ("baseline", "transfer"):
        rs = [rec[(n, s, arm)] for s in range(10) if (n, s, arm) in rec]
        if not rs:
            continue
        agg[(n, arm)] = dict(
            s1_r2=np.mean([r["S1_eV"]["r2"] for r in rs]), s1_rho=np.mean([r["S1_eV"]["rho"] for r in rs]),
            sg_r2=np.mean([r["Sigma_opt"]["r2"] for r in rs]), sg_rho=np.mean([r["Sigma_opt"]["rho"] for r in rs]),
            s1_mae=np.mean([r["S1_eV"]["mae"] for r in rs]), sg_mae=np.mean([r["Sigma_opt"]["mae"] for r in rs]),
            k=len(rs))
for n in sizes:
    b, t = agg.get((n, "baseline")), agg.get((n, "transfer"))
    f = lambda a: (f"{a['s1_r2']:+.3f}/{a['s1_rho']:+.2f} | {a['sg_r2']:+.3f}/{a['sg_rho']:+.2f} | "
                   f"{a['s1_mae']:.3f}/{a['sg_mae']:.3f}") if a else "—"
    print(f"{n:>3} | {f(b):>40} | {f(t):>40}")

print("\n配对 Δ（test7，同划分 5 个 seed，transfer − baseline）:")
for n in sizes:
    d_s1 = [rec[(n, s, "transfer")]["S1_eV"]["r2"] - rec[(n, s, "baseline")]["S1_eV"]["r2"]
            for s in range(10) if (n, s, "transfer") in rec and (n, s, "baseline") in rec]
    d_sg = [rec[(n, s, "transfer")]["Sigma_opt"]["r2"] - rec[(n, s, "baseline")]["Sigma_opt"]["r2"]
            for s in range(10) if (n, s, "transfer") in rec and (n, s, "baseline") in rec]
    if not d_s1:
        continue
    T = {4: 2.776, 9: 2.262, 5: 2.571}.get(len(d_s1) - 1, 2.776)
    for nm, dd in (("S1", d_s1), ("σ", d_sg)):
        a = np.array(dd); mu, sd = a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0)
        ci = T * sd / np.sqrt(len(a))
        tag = "迁移↑显著" if mu - ci > 0 else ("基线↑显著" if mu + ci < 0 else "跨0")
        print(f"  n={n:<3} Δ{nm:<2} R² = {mu:+.3f} ± {sd:.3f} (CI±{ci:.3f}) {tag}  "
              f"[{sum(1 for x in a if x>0)}/{len(a)} 为正]")

print("\n两个实验分子在 test7 上的预测（各规模各臂均值，eV / nm）:")
for nm, (s1, sg, wl) in EXP.items():
    print(f"  {nm}: 实验/DFT 参考 S1={s1:.3f} eV ({wl:.1f} nm), σ={sg:.3f}")
    for n in sizes:
        cells = []
        for arm in ("baseline", "transfer"):
            pr = [rec[(n, s, arm)]["pred"][nm] for s in range(10) if (n, s, arm) in rec and nm in rec[(n, s, arm)]["pred"]]
            if pr:
                a = np.array(pr)
                cells.append(f"{arm[:4]}: {a[:,0].mean():.3f} eV ({HC/a[:,0].mean():.0f} nm) σ={a[:,1].mean():.3f}")
        print(f"     n={n:<3} " + " | ".join(cells))
json.dump({f"{k[0]}|{k[1]}": v for k, v in agg.items()},
          open(f"{OUT.rsplit('/',1)[0]}/test7b_summary.json", "w"), indent=2, ensure_ascii=False)
json.dump({f"{k[0]}|{k[1]}|{k[2]}": {t: v[t] for t in ("S1_eV", "Sigma_opt")} for k, v in rec.items()},
          open(f"{OUT.rsplit('/',1)[0]}/test7b_per_run.json", "w"), indent=2, ensure_ascii=False)
print("\n→ report/test7b_summary.json + test7b_per_run.json")
