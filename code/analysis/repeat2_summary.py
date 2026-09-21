#!/usr/bin/env python3
"""重训后（σ 口径修正）的 valid 口径统计：10 组划分 × 5 规模 × 2 臂
输出: report/repeat2_valid_summary.json + 屏幕表格（含配对 Δ 与 95%CI）
"""
import glob, json, os, re
import numpy as np

W = "/mnt/f/green-laser/3core"
SIZES = [6, 13, 27, 40, 53]
T95 = {3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}
EP = re.compile(r"epoch (\d+) train loss")
M0 = re.compile(r"target_0\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
M1 = re.compile(r"target_1\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")
AG = re.compile(r"AGGREGATE\s+MAE\s+([\d.]+)\s+R²\s+(-?[\d.]+)")


def parse(log):
    rows, cur = [], {}
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        if EP.search(line):
            cur = {"epoch": int(EP.search(line).group(1))}
        a, b, c = M0.search(line), M1.search(line), AG.search(line)
        if a and cur:
            cur["s1_mae"], cur["s1_r2"] = float(a.group(1)), float(a.group(2))
        if b and cur:
            cur["sg_mae"], cur["sg_r2"] = float(b.group(1)), float(b.group(2))
        if c and cur:
            cur["agg"], cur["tot_r2"] = float(c.group(1)), float(c.group(2))
            rows.append(cur); cur = {}
    return rows


per = {}
for ck in sorted(glob.glob(f"{W}/runs/repeat2/tr*_s*/*/save_finetune/train.log")):
    d = os.path.dirname(ck)
    arm = os.path.basename(os.path.dirname(d))
    sid = os.path.basename(os.path.dirname(os.path.dirname(d)))
    n = int(sid.split("_")[0][2:]); s = int(sid.split("_s")[1])
    rows = [r for r in parse(ck) if "tot_r2" in r]
    if not rows:
        print("  ⚠ 无指标:", ck); continue
    best = min(rows, key=lambda r: r["agg"])
    per[f"{n}|{s}|{arm}"] = best
json.dump(per, open(f"{W}/report/repeat2_per_run.json", "w"), indent=1, ensure_ascii=False)
print(f"解析 run: {len(per)}/100\n")

out = {}
print(f"{'n':>3} {'臂':<9}{'agg_mae':>9}{'S1 MAE':>9}{'S1 R²':>8}{'σ MAE':>9}{'σ R²':>8}{'总 R²':>8}")
for n in SIZES:
    for arm in ("baseline", "transfer"):
        v = [per[f"{n}|{s}|{arm}"] for s in range(10) if f"{n}|{s}|{arm}" in per]
        m = {k: float(np.mean([x[k] for x in v])) for k in ("agg", "s1_mae", "s1_r2", "sg_mae", "sg_r2", "tot_r2")}
        sd = {k: float(np.std([x[k] for x in v], ddof=1)) for k in ("agg", "s1_mae", "s1_r2", "sg_mae", "sg_r2", "tot_r2")}
        out[f"{n}|{arm}"] = {"mean": m, "sd": sd, "n_splits": len(v)}
        print(f"{n:>3} {arm:<9}{m['agg']:>9.3f}{m['s1_mae']:>9.3f}{m['s1_r2']:>8.3f}{m['sg_mae']:>9.3f}{m['sg_r2']:>8.3f}{m['tot_r2']:>8.3f}")

print(f"\n配对 Δ（迁移 − 基线，同划分；10 组划分的均值±SD 与 95%CI）:")
print(f"{'n':>3} {'Δ总R²':>18}{'95%CI':>10}{'为正':>6}   {'ΔσR²':>18}{'ΔS1MAE':>18}")
for n in SIZES:
    ks = [s for s in range(10) if f"{n}|{s}|transfer" in per]
    for key in ("tot_r2", "sg_r2", "s1_mae"):
        pass
    d_tot = np.array([per[f"{n}|{s}|transfer"]["tot_r2"] - per[f"{n}|{s}|baseline"]["tot_r2"] for s in ks])
    d_sg = np.array([per[f"{n}|{s}|transfer"]["sg_r2"] - per[f"{n}|{s}|baseline"]["sg_r2"] for s in ks])
    d_s1m = np.array([per[f"{n}|{s}|baseline"]["s1_mae"] - per[f"{n}|{s}|transfer"]["s1_mae"] for s in ks])
    ci = T95.get(len(ks) - 1, 2.262) * d_tot.std(ddof=1) / np.sqrt(len(ks))
    out[f"{n}|delta"] = {"tot_r2_mean": float(d_tot.mean()), "tot_r2_sd": float(d_tot.std(ddof=1)),
                         "ci95": float(ci), "positive": int((d_tot > 0).sum()),
                         "sg_r2_mean": float(d_sg.mean()), "s1_mae_improve_mean": float(d_s1m.mean())}
    print(f"{n:>3} {d_tot.mean():>+10.3f}±{d_tot.std(ddof=1):<6.3f}{'±'+format(ci,'.3f'):>10}{(d_tot>0).sum():>4}/10"
          f"   {d_sg.mean():>+10.3f}±{d_sg.std(ddof=1):<6.3f}{d_s1m.mean():>+12.3f}±{d_s1m.std(ddof=1):<6.3f}")
json.dump(out, open(f"{W}/report/repeat2_valid_summary.json", "w"), indent=1, ensure_ascii=False)
print(f"\n→ {W}/report/repeat2_valid_summary.json")
