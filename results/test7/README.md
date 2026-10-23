# results/test7 — 7 个锁定独立测试分子上的逐 run 预测（100 run）

每个 CSV = 一次 run（`tr{n}_s{seed}_{arm}.csv`），7 行 = 7 个锁定分子。
`target_0` = S₁ 激发态能量（eV），`target_1` = 净发射截面 σopt（10⁻¹⁶ cm²）。

## ⚠ 2026-10-23 的真值列修正（重要）

- 本目录最初（2026-09-19）由 `test7b` 评估产出：`target_1_pred` 来自**修正后**模型，
  但 `target_1_true` 仍是 **σ 修正前**的旧标签（旧值 = 现值 ÷ π）。
  两者差 π ⇒ 当时算出的 σopt MAE/R² 无效（R²≈−50~−82）。
- 现 `target_1_true` 已按 `data/labels/labels_69.csv`（现行口径，含 4π）覆写，
  逐行校验 `旧值 × π == 新值`：700/700 行吻合（最大偏差 1.2e-6，纯四舍五入）。
- `target_0_*`（S₁）与 `target_1_pred`（σ 预测）**未做任何改动**；
  模型训练与预测本身没有问题，错的只是这一列真值的版本。
- 可复现修正过程：`/home/sakura/app/anaconda3/bin/python3 code/analysis/fix_test7_sigma_labels.py`
- 重算指标：`results/summary/test7_summary.json`（旧口径的错误汇总在
  `results/summary/_deprecated_prefix-labels/` 留档）。
- 原始未修正副本（审计用，不在仓库内）：`E:\paper-lasergreen\version2\_bak_test7_prepi_20261023\`

## 读法

- S₁（target_0）：`target_0_mae` 是最稳的指标（7 个点上 R² 不稳定，勿作主指标）
- σopt（target_1）：修正后 MAE 0.52–0.99；R² 在 7 点上波动大，仅作参考
- 锁定测试集只有 7 个点 ⇒ 排序类指标（Spearman）噪声大，需与 10 组划分的验证集结果一起看
