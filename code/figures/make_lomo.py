#!/usr/bin/env python3
"""留一模组（LOMO）划分 + task spec：
  LOCO 留一核心（Py/TT/Th）、LOBO 留一桥连（≥5 个分子）、LOTO 留一端基（≥5 个分子）
  每个 fold：test = 该模组全部分子；train/valid = 其余按 8:2（seed 0，仅用于早停）
"""
import csv, json, os
from collections import Counter, defaultdict

P = "/mnt/i/green-laser-3core"
W = "/mnt/f/green-laser/3core"
rows = list(csv.DictReader(open(f"{P}/computed/labels_69.csv")))
names = [r["name"] for r in rows]

folds = []
for kind, idx in (("LOCO", 0), ("LOBO", 1), ("LOTO", 2)):
    groups = defaultdict(list)
    for r in rows:
        p = r["name"].split("_")
        key = p[idx] if len(p) > idx + 1 else None   # 2 段名（无桥）跳过桥维
        if kind == "LOTO":
            key = p[-1]
        if key:
            groups[key].append(r["name"])
    for g, ms in sorted(groups.items()):
        if kind != "LOCO" and len(ms) < 5:      # 桥/端基只取 ≥5 个的组
            continue
        if kind == "LOCO" and len(ms) < 5:
            continue
        folds.append((f"{kind}_{g}", ms))

os.makedirs(f"{P}/splits_lomo", exist_ok=True)
manifest = []
made = 0
for tag, ms in folds:
    test = set(ms)
    EXP = {"Th_Th_Fl", "Py_dTT_Fl"}          # 两个实验分子永不出现在训练/验证中
    rest = [n for n in names if n not in test and n not in EXP]
    perm = sorted(rest)
    n_tr = int(round(len(perm) * 0.8))
    train, valid = perm[:n_tr], perm[n_tr:]
    for fn, lst in ((f"{tag}.csv", sorted(test | set(train))), (f"{tag}_valid.csv", valid)):
        with open(f"{P}/splits_lomo/{fn}", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["name", "S1_eV", "Sigma_opt", "Peak_WL_nm", "role"])
            d = {r["name"]: r for r in rows}
            for n in lst:
                r = d[n]
                w.writerow([n, r["S1_eV"], r["Sigma_opt"], r["Peak_WL_nm"], r["role"]])
    manifest.append({"fold": tag, "n_test": len(test), "n_train": len(train), "n_valid": len(valid),
                     "test_group": sorted(test)})
    # 只让 train+valid 进训练数据：prepare 用 csv=test+train 会漏 valid → 直接写 train+valid
    with open(f"{P}/splits_lomo/{tag}_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "S1_eV", "Sigma_opt", "Peak_WL_nm", "role"])
        d = {r["name"]: r for r in rows}
        for n in sorted(set(train) | set(valid)):
            r = d[n]
            w.writerow([n, r["S1_eV"], r["Sigma_opt"], r["Peak_WL_nm"], r["role"]])
    for arm, ck in (("transfer", f"{P}/source/checkpoint/checkpoint_best.pt"),
                    ("baseline", f"{P}/source/official_weights/mol_pre_all_h_220816.pt")):
        wd = f"{W}/runs/lomo/{tag}/{arm}"
        spec = {"name": f"3core-lomo-{tag}-{arm}",
                "input": {"MOLS_DIR": P, "TRAIN_ROOT": W, "FOLD": tag, "ARM": arm, "CKPT": ck},
                "steps": [
                    {"id": "prep", "tool": "unimol", "params": {
                        "subcommand": "prepare_unicore",
                        "csv": "$input.MOLS_DIR/splits_lomo/$input.FOLD_all.csv",
                        "valid_csv": "$input.MOLS_DIR/splits_lomo/$input.FOLD_valid.csv",
                        "xyz_dir": "$input.MOLS_DIR/computed/xyz", "name_col": "name",
                        "target_cols": "S1_eV;;Sigma_opt", "work_dir": wd,
                        "random_state": 0, "task_name": "laser"}},
                    {"id": "train", "tool": "unimol", "depends_on": ["prep"], "params": {
                        "subcommand": "train_local", "work_dir": wd,
                        "finetune_from_model": "$input.CKPT", "task_name": "laser", "num_classes": 2,
                        "max_epoch": 200, "batch_size": 16, "lr": "1e-4", "encoder_layers": "15",
                        "dropout": "0.1", "only_polar": -1, "conf_size": 1, "device": "cuda",
                        "num_workers": 0, "conda_env": "py39-unimol", "dict_name": "pretrain_dict.txt"}}]}
        json.dump(spec, open(f"{P}/specs/lomo_{tag}_{arm}.json", "w"), indent=2, ensure_ascii=False)
        made += 1
json.dump(manifest, open(f"{P}/splits_lomo/manifest.json", "w"), indent=2, ensure_ascii=False)
print(f"LOMO folds: {len(folds)} 个（{made} 个 spec）")
for m in manifest:
    print(f"  {m['fold']:<12} test {m['n_test']:>3} 个 | train {m['n_train']} | valid {m['n_valid']}")
