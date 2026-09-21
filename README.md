# Transfer-Learning-Driven Targeted Discovery of Green Organic Laser Molecules

Data and code for the manuscript *"迁移学习驱动的高性能有机绿色激光分子定向发现"*
(*Acta Chimica Sinica*). This repository contains everything needed to reproduce the
training, evaluation and screening results, **except** model weights (not released) and
ASE experimental raw spectra (available from the authors on request).

---

## 1. Environment

| Component | Version / setting |
|---|---|
| DFT | ORCA 6.0.1, B3LYP-D3BJ/def2-SVP, RIJCOSX + def2/J, gas phase, charge 0 / mult 1 |
| TD-DFT | S0 geometry: NRoots = 10 (absorption); S1 geometry: NRoots = 3 (optimization) / 30 (emission) |
| Conformer pre-optimization | xtb, GFN2-xTB (lowest-energy conformer of all cis/trans isomers) |
| Re-absorption / ESA | Multiwfn (S0→Sn absorption, S1→Sn excited-state absorption) |
| Net emission cross-section | σ(λ) = e²λ²/(4πε₀mₑc₀²n_F)·g(λ)·f, with Gaussian FWHM 75 nm (emission) / 125 nm (absorption, re-absorption) and n_F = 1.75; σ_opt = σ_em − σ_S0 − σ_S1 |
| Model | Uni-Mol (15 encoder layers, 512-dim, 64 heads, dropout 0.1) |
| Training | Python 3.9, PyTorch, Adam (lr 1e-4, β = 0.9/0.99), batch size 16, polynomial decay, warmup 0.06, fp16, early stopping on validation `agg_mae` |
| Hardware | 1 × NVIDIA RTX 4060 Ti (16 GB) |

Python packages: see `requirements.txt`.

---

## 2. Repository layout

```
code/
  model/            no-unicore Uni-Mol training / inference scripts
  analysis/         split generation, statistics, SA, cross-section recomputation, ranking
  figures/          scripts generating every figure of the manuscript and the SI
  run_scripts/      batch-orchestration examples (one window = one batch)
configs/
  specs/            106 training/inference configs (the 100 runs reported in Table 1 + inference)
  tasks_examples/   examples of the executable task format
data/
  splits/           103 split files (see §3.1)
  labels/           labels_69.csv, pool_67.csv, locked_2.csv
  library/          library_3core.csv (330 candidates + building-block annotation), library_330_SA.csv
  xyz/              330 xtb-optimized conformers (the DFT starting coordinates)
  dft_tables/       absorption.csv, esa.csv, emission.csv, sigma_summary.csv (75 molecules)
results/
  valrank/          100 per-run per-sample predictions on the held-out validation sets
  test7/            100 per-run per-sample predictions on the locked 7-molecule test set
  predictions/      predictions for all 330 candidates
  summary/          aggregated statistics (JSON)
seeds.txt           random-seed policy
```

---

## 3. Data dictionary

### 3.1 `data/splits/*.csv`
One file per (training size × random split), 103 files. Columns:

| column | meaning |
|---|---|
| `name` | molecule ID (`Core_Bridge_Terminal`, e.g. `Th_Th_Fl` = thiophene core / terthiophene bridge / fluorene terminal) |
| `S1_eV` | S1 excited-state energy from TD-DFT (eV) — regression target 0 |
| `Sigma_opt` | net emission cross-section (10⁻¹⁶ cm²) — regression target 1 |
| `Peak_WL_nm` | emission wavelength (nm) |
| `role` | `train` / `valid` / `test` |

Split policy: the 62 pool molecules are split into `train` (n = 6, 13, 27, 40 or 53) and
`valid` (62 − n); 7 molecules are held out as a fixed `test` set (including the two
experimentally validated materials FL3T and PyDTTFL) and are never used for training,
hyper-parameter search or checkpoint selection.

### 3.2 `data/labels/`
* `labels_69.csv` — the 69 target-domain molecules with DFT labels (matching `data/splits`).
* `pool_67.csv` — the 67-molecule pool (69 minus the 2 experimental molecules).
* `locked_2.csv` — the 2 experimental molecules.

### 3.3 `data/library/`
* `library_3core.csv` — all 330 candidates (3 cores × 10 bridges × 10 terminals + 10 bridge-free).
* `library_330_SA.csv` — `name, SMILES, SA` (RDKit `sascorer` on the H-removed molecule).

### 3.4 `data/dft_tables/`
| file | content |
|---|---|
| `absorption.csv` | S0→Sn TD-DFT transitions (75 molecules × 10 roots): wavelength, oscillator strength |
| `esa.csv` | S1→Sn excited-state absorption (75 molecules × 30 roots) |
| `emission.csv` | S1 emission (75 molecules): wavelength, oscillator strength |
| `sigma_summary_75molecules.csv` | σ_em, σ_S0, σ_S1 and σ_opt per molecule |

### 3.5 `results/valrank/*.csv`
Per-sample predictions on the held-out validation set, one file per run
(`tr{n}_s{seed}_{arm}.csv`, `arm ∈ {transfer, baseline}`):

| column | meaning |
|---|---|
| `name` | molecule ID (`*_valid` suffix) |
| `target_0_true` / `target_0_pred` | S1 energy: reference (eV) / model output (standardized) |
| `target_1_true` / `target_1_pred` | σ_opt: reference (10⁻¹⁶ cm²) / model output (standardized) |

Model outputs are z-scored; reference values are in physical units. Rank-based metrics
(Spearman ρ) are unaffected by this monotone transform; to recover physical predictions use

```
S1_eV  = pred_0 * 0.19435146020749233 + 2.3972511509433962
sigma  = pred_1 * 1.304328990555548   + 3.2939660000000006
```

(the same normalization constants the models were trained with, stored as
`property_norms.json` next to each checkpoint).

---

## 4. Reproducing the results

```bash
# 0) splits are provided as CSV files in data/splits/ (see seeds.txt for the seed index).
#    To regenerate: shuffle the 62 pool molecules with numpy.random.default_rng(seed)
#    and take the first n as train, the rest as valid (the 7 locked molecules stay untouched).

# 1) train — one run per (training size, split seed, arm); configs are identical except
#    the encoder initialization (transfer = source-domain checkpoint, baseline = random init)
python code/model/train_no_unicore.py <work_dir> --task-name laser --task mol_finetune \
        --loss finetune_smooth_mae --arch unimol_base --num-classes 2 --encoder-layers 15 \
        --lr 1e-4 --batch-size 16 --max-epoch 200 --seed 42 --patience 20 \
        --best-checkpoint-metric valid_agg_mae --finetune-from-model <source_checkpoint>

# 2) evaluate on the held-out validation set / locked test set (per-sample predictions)
python code/model/infer_no_unicore.py <mirror_dir> --split valid --checkpoint <ckpt> \
        --output results/valrank/<run>.csv

# 3) aggregate — Table 1 statistics, Spearman rho, binned MAE
python code/analysis/repeat2_summary.py     # Table 1 (10 splits, paired delta, 95% CI)
python code/analysis/valrank_stats.py       # Spearman rho on the held-out validation sets
python code/analysis/test7b_stats.py        # locked independent test set
```

`baseline` runs use the same configuration with a randomly initialized encoder
(no pre-trained weights loaded). Every reported number is the mean over 10 independent
splits; paired differences Δ = transfer − baseline are reported with 95 % confidence intervals.

---

## 5. Random-seed policy

* Split seeds: `0–9` (10 independent random splits per training size) — see `seeds.txt`.
* Training seed: `42` (identical for both arms and all sizes).
* No other source of randomness is used; DataLoader worker count is 0.

---

## 6. License

MIT — see `LICENSE`.

## 7. Citation

If you use these data or code, please cite the manuscript:
> Wei Y.-S.; Ding Y.-T.; Feng M.-H.; Zhang Z.; Zhao G.; Gao Z.; Xu K.; Huo Y.*; Zhang H.-L.*
> *Acta Chim. Sinica* (in press).
