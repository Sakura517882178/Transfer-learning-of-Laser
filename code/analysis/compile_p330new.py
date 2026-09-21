#!/usr/bin/env python3
"""重训后 330 全库预测整理：合并 → 精度核验（对 69 个已知 DFT）→ SA/FOM → 排序表
输出: report/p330new_all.csv, report/p330new_FOM.csv
"""
import csv
import numpy as np
from scipy.stats import spearmanr, pearsonr

REP = "/mnt/f/green-laser/3core/report"
P = "/mnt/i/green-laser-3core"
HC = 1239.84193

pred = {}
for f in ("p330n_part1.csv", "p330n_part2.csv"):
    for r in csv.DictReader(open(f"{REP}/{f}")):
        n = r["name"].replace("_valid", "").replace("_train", "")
        pred[n] = (float(r["target_0_pred"]), float(r["target_1_pred"]))
print(f"合并: {len(pred)} 个分子")

lab = {r["name"]: (float(r["S1_eV"]), float(r["Sigma_opt"])) for r in
       csv.DictReader(open(f"{P}/computed/labels_69.csv"))}


def conv(n):
    p_ = n.split("_")
    if p_[-2:] == ["Fl", "Ph"]:
        t, b = "PhFl", (p_[1] if len(p_) >= 3 else "")
    else:
        t, b = p_[-1], (p_[1] if len(p_) >= 3 else "")
    b = "Ph" if b == "P" else b
    return f"{p_[0]}_{t}" if not b else f"{p_[0]}_{b}_{t}"


lab2 = {conv(k): v for k, v in lab.items()}
common = [n for n in pred if n in lab2]
p1 = np.array([pred[n][0] for n in common]); t1 = np.array([lab2[n][0] for n in common])
p2 = np.array([pred[n][1] for n in common]); t2 = np.array([lab2[n][1] for n in common])
print(f"精度核验（{len(common)} 个有 DFT 参考）:")
print(f"  S1   : MAE = {np.abs(p1-t1).mean():.4f} eV | ρ = {spearmanr(t1,p1).correlation:+.3f} | R² = {pearsonr(t1,p1)[0]**2:.3f}")
print(f"  σopt : MAE = {np.abs(p2-t2).mean():.4f} | ρ = {spearmanr(t2,p2).correlation:+.3f} | R² = {pearsonr(t2,p2)[0]**2:.3f}")

# 全库 + SA + FOM
sa = {r["name"]: float(r["SA"]) for r in csv.DictReader(open(f"{P}/library/library_330_SA.csv")) if r["SA"]}
rows = []
for n, (s1, sg) in pred.items():
    wl = HC / s1 if s1 > 0 else None
    s = sa.get(n)
    rows.append({"name": n, "s1": s1, "wl": wl, "sg": sg, "sa": s, "fom": (sg / s if s else None),
                 "known": n in lab2})
with open(f"{REP}/p330new_all.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["name", "S1_eV_pred", "lambda_nm_pred", "Sigma_opt_pred", "SA", "FOM_sigma_SA", "has_DFT"])
    for r in sorted(rows, key=lambda x: -x["sg"]):
        w.writerow([r["name"], f"{r['s1']:.4f}", f"{r['wl']:.1f}", f"{r['sg']:.4f}",
                    f"{r['sa']:.3f}" if r["sa"] else "", f"{r['fom']:.4f}" if r["fom"] else "",
                    "yes" if r["known"] else "no"])
green = sorted([r for r in rows if r["wl"] and r["wl"] > 500 and r["fom"]], key=lambda r: -r["fom"])
rank = {r["name"]: i for i, r in enumerate(green, 1)}
with open(f"{REP}/p330new_FOM.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["rank_FOM_green", "name", "lambda_nm_pred", "Sigma_opt_pred", "SA", "FOM_sigma_SA", "has_DFT"])
    for r in green:
        w.writerow([rank[r["name"]], r["name"], f"{r['wl']:.1f}", f"{r['sg']:.4f}", f"{r['sa']:.3f}", f"{r['fom']:.4f}",
                    "yes" if r["known"] else "no"])
print(f"\n绿光区（λ>500 nm）{len(green)} 个，FOM = σopt/SA 前 10：")
print(f"{'排名':>4} {'分子':<18}{'λ_pred':>8}{'σopt':>8}{'SA':>7}{'FOM':>8}  DFT")
for r in green[:10]:
    print(f"{rank[r['name']]:>4} {r['name']:<18}{r['wl']:>7.0f}n{r['sg']:>8.2f}{r['sa']:>7.2f}{r['fom']:>8.3f}  {'有' if r['known'] else '—'}")
print("\n两个实验分子（新模型预测 vs DFT 参考）：")
for tgt in ("Th_Th_Fl", "Py_dTT_Fl"):
    r = [x for x in rows if x["name"] == tgt][0]
    ref = lab2.get(tgt)
    rk = rank.get(tgt)
    print(f"  {tgt:<12} 预测 λ={r['wl']:.0f} nm σopt={r['sg']:.3f} | DFT σopt={ref[1]:.3f} (λ={HC/ref[0]:.0f} nm) | "
          f"绿光区 FOM 排名 {rk}/{len(green)}" if rk else f"  {tgt}: 排名 —")
print(f"\n→ {REP}/p330new_all.csv, {REP}/p330new_FOM.csv")
