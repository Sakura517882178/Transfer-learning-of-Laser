# 已废弃：这些文件的口径有误（勿引用）

2026-10-23 核实：`test7b_summary.json` / `test7b_per_run.json` 是在
`results/test7/*.csv` 的 `target_1_true`（σopt 真值）仍为 **2026-09-19 σ 修正前**旧标签
（= 修正值 ÷ π）时算出来的。同文件的 `target_1_pred` 已来自修正后模型（尺度正确），
两者差 π，故其中的 **σopt 的 MAE/R² 全是假的**（R²≈−50~−82）。S₁ 的指标不受影响。

处置（2026-10-23，可回滚）：
- `results/test7/*.csv` 的 `target_1_true` 已用 `data/labels/labels_69.csv` 的现行值覆盖
  （逐行校验 `旧值 × π == 新值`，700/700 行吻合，最大偏差 1.2e-6）；
  `target_0_*`（S₁ 真值/预测）与 `target_1_pred`（σ 预测）**未改动**。
- 重算后的指标见 `results/summary/test7_summary.json` / `test7_per_run.json`。
- 训练/推理记录仍保留在计算目录 `F:\green-laser\3core\report\test7b\`（历史原始件，未改动）。

本目录仅作审计留档。
