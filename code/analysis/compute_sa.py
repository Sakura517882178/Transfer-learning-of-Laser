#!/usr/bin/env python3
"""330 个候选分子：xyz → 键级判定 → SMILES → 合成可行性 SA（Ertl & Schuffenhauer）
输出: /mnt/i/green-laser-3core/library/library_330_SA.csv  (name, smiles, SA, n_atoms, status)
"""
import csv, glob, os, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds, RDConfig

sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402

RDLogger.DisableLog("rdApp.*")
XYZ = "/mnt/i/green-laser-3core/library/xyz"
OUT = "/mnt/i/green-laser-3core/library/library_330_SA.csv"

rows, ok, fail = [], 0, []
for f in sorted(glob.glob(f"{XYZ}/*.xyz")):
    name = os.path.basename(f)[:-4]
    try:
        mol = Chem.MolFromXYZFile(f)
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        smi = Chem.MolToSmiles(mol)
        sa = sascorer.calculateScore(mol)
        rows.append([name, smi, f"{sa:.3f}", mol.GetNumAtoms(), "ok"])
        ok += 1
    except Exception as e:
        rows.append([name, "", "", "", f"fail:{type(e).__name__}"])
        fail.append((name, type(e).__name__))
with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["name", "smiles", "SA", "n_atoms", "status"])
    w.writerows(rows)
print(f"成功 {ok}/{len(rows)} → {OUT}")
if fail:
    print("失败样例:", fail[:8])
