# 2026-08-14 实验可视化长期约束

用户要求：每次实验结束必须同时生成一张能够目视检查“预测值 vs 真实值全览”的图。

## 固定内容

使用 `Mooney_Prediction_Pipeline/reporting/generate_experiment_fullview_plot_v1.py` 生成 PNG，至少包含：

1. OOT 实际 MNY vs 预测 MNY 散点图，带 `y=x`、±2 MU、±5 MU 参考线。
2. 按时间排序的 OOT 实际/预测全览曲线。
3. T15760 试点单独散点图（若当前实验包含该配方）。
4. OOT 残差分布。
5. 图下注明有效样本数、覆盖率、MAE、Bias、预测Std/实际Std，并明确当前 Lab/未来组是否排除。

## 文件约定

- 图和指标放在同一个实验目录。
- PNG 命名：`<experiment>_actual_vs_predicted_fullview.png`。
- 同时保存 `<experiment>_plot_metrics.json`。
- 图中只使用经过正式评价口径处理后的组级数据；不能把同一 Lab 值按车次重复计权。
- 如果模型因缺少前序状态而阻断，图中必须显示有效覆盖率，而不能把阻断行伪造为预测值。

## 本次执行

- Delta JIT 图：`reports/v371_challenger/global_rds_active_v1/m1_order_delta_residual_v1/m1_order_delta_jit_actual_vs_predicted_fullview.png`
- OOT 有效预测：37/211（17.5%）。
- OOT MAE：5.33 MU；Bias：0.37 MU；预测Std/实际Std：1.081。
