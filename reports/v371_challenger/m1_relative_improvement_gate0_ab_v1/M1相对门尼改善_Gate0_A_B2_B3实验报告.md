# M1 相对门尼改善 Gate 0 / A / B2 / B3 实验报告

## 结论

- Gate 0 共审计 1127 个精确控制杠杆，`CAUSAL_ACTION_SUPPORTED` 为 0 个。
- 方案 A 建立了去混杂历史包络，但统一保持 `HISTORICAL_ENVELOPE_SHADOW_ONLY`，不声明可降低具体 MU。
- B2 方向/幅度门禁：`False`。
- B3：T15760=`CONTROL_TRANSFER_UNAVAILABLE`，B00458=`CONTROL_TRANSFER_UNAVAILABLE`。
- B4：`BLOCKED_NO_CAUSAL_ACTION_SUPPORT`；B5：`BLOCKED_PREREQUISITE_GATES`。
- B6 尚未开始，因此 B7 为 `BLOCKED_DUAL_PILOT_GATE`。

## Gate 0：自然动作支持

| compound | causal_support_status | size |
| --- | --- | --- |
| M1-B00458---- 12 005 | INSUFFICIENT_LAB_SUPPORT | 21 |
| M1-B00458---- 12 005 | LOCKED_EXPERT_RULE | 12 |
| M1-T15760---- 06 002 | FIXED_IN_HISTORY | 338 |
| M1-T15760---- 06 002 | INSUFFICIENT_LAB_SUPPORT | 344 |
| M1-T15760---- 06 002 | LOCKED_EXPERT_RULE | 412 |

没有任何杠杆满足“至少30个独立Lab组转移、至少3个主要控制水平、每个主要水平至少8组”的联合条件。B00458旧Step快照与当前Pallet标签快照没有物理键交集，因此没有把单批次伪装成Lab组。

## 方案 A：去混杂历史包络

- 静态/原料/环境基线 OOT MAE：5.246 MU。
- OOT Bias：-1.073 MU。
- 满足当前统计条件的包络：7；全部仍为影子关联证据，不转为因果动作。

| TemplateKey | groups | envelope_status | raw_low_groups | envelope_groups | raw_p20 | adjusted_p20 | raw_low_effect_mu | adjusted_low_effect_mu | advantage_attenuation | stratum_direction_consistency | rolling_windows_with_low_direction | adjusted_effect_bootstrap_p05 | adjusted_effect_bootstrap_p95 | dominant_material_environment_stratum_share | confounding_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| b3e1e3a908eecaca | 76 | HISTORICAL_ENVELOPE_SUPPORTED | 16.0000 | 15.0000 | 53.9800 | -6.0799 | -10.2500 | -10.2588 | -0.0009 | 1.0000 | 3.0000 | -12.5258 | -8.2241 | 0.4375 | NO_DOMINANT_MEASURED_CONFOUNDING |
| 0c2d07bd4d95f445 | 53 | HISTORICAL_ENVELOPE_SUPPORTED | 11.0000 | 9.0000 | 55.1620 | -3.7255 | -8.0850 | -8.1167 | -0.0039 | 1.0000 | 2.0000 | -9.5431 | -5.3913 | 0.4545 | NO_DOMINANT_MEASURED_CONFOUNDING |
| 2f462988f4ff8440 | 41 | HISTORICAL_ENVELOPE_SUPPORTED | 9.0000 | 8.0000 | 60.4400 | -4.7886 | -10.8100 | -10.3543 | 0.0422 | 1.0000 | 3.0000 | -15.3921 | -7.6157 | 0.4444 | NO_DOMINANT_MEASURED_CONFOUNDING |
| 427625bd42143962 | 35 | HISTORICAL_ENVELOPE_SUPPORTED | 7.0000 | 7.0000 | 56.9380 | -5.5018 | -8.3100 | -9.3109 | -0.1205 | 1.0000 | 2.0000 | -15.2391 | -6.3840 | 0.4286 | NO_DOMINANT_MEASURED_CONFOUNDING |
| a5cc1cea5a28cfcc | 24 | HISTORICAL_ENVELOPE_SUPPORTED | 5.0000 | 5.0000 | 52.2400 | -1.8569 | -3.0800 | -3.1000 | -0.0065 | 1.0000 | 2.0000 | -5.1503 | -2.0817 | 0.4000 | NO_DOMINANT_MEASURED_CONFOUNDING |
| aefba841c8fee56b | 23 | HISTORICAL_ENVELOPE_SUPPORTED | 5.0000 | 5.0000 | 54.6580 | -3.3177 | -9.4500 | -9.7468 | -0.0314 | 1.0000 | 2.0000 | -13.9841 | -5.5368 | 0.6000 | NO_DOMINANT_MEASURED_CONFOUNDING |
| dc987655b6c0229d | 20 | HISTORICAL_ENVELOPE_SUPPORTED | 4.0000 | 3.0000 | 43.6500 | -1.9586 | -2.8500 | -2.5529 | 0.1042 | 1.0000 | 2.0000 | -4.1058 | -1.5398 | 0.5000 | NO_DOMINANT_MEASURED_CONFOUNDING |
| b232fa0e1be2328a | 19 | INSUFFICIENT_TEMPLATE_HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |
| c988e732a247078d | 18 | INSUFFICIENT_TEMPLATE_HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 260ebfb4da63ce0d | 17 | INSUFFICIENT_TEMPLATE_HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |
| 400bb896debeb7dc | 17 | INSUFFICIENT_TEMPLATE_HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |
| cb559ae0b7e32162 | 12 | INSUFFICIENT_TEMPLATE_HISTORY |  |  |  |  |  |  |  |  |  |  |  |  |  |

## B2：方向与幅度模型

| split | scope | pairs | significant_pairs | direction_accuracy_abs_actual_ge_2 | reduction_recall | delta_mae | actual_reduction_std | predicted_reduction_std | delta_std_ratio | p10_p90_coverage |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | GLOBAL_M1 | 147 | 103 | 0.8835 | 0.8861 | 4.6220 | 9.5839 | 1.7270 | 0.1802 | 0.7823 |
| train | T15760 | 82 | 64 | 0.8906 | 0.8333 | 5.5309 | 10.3139 | 1.9245 | 0.1866 | 0.7317 |
| val | GLOBAL_M1 | 36 | 29 | 0.4138 | 0.3571 | 6.1121 | 7.8198 | 1.3786 | 0.1763 | 0.6667 |
| val | T15760 | 28 | 23 | 0.4348 | 0.3636 | 6.2898 | 7.9246 | 1.2442 | 0.1570 | 0.6786 |
| val | B00458 | 1 | 0 |  | 0.0000 | 1.2481 | 0.0000 | 0.0000 |  | 1.0000 |
| oot | GLOBAL_M1 | 29 | 21 | 0.7619 | 0.6000 | 4.9176 | 6.9735 | 1.5987 | 0.2292 | 0.7241 |
| oot | T15760 | 21 | 16 | 0.8125 | 0.6000 | 4.6948 | 7.2950 | 1.7012 | 0.2332 | 0.7143 |
| oot | B00458 | 1 | 0 |  | 0.0000 | 3.1490 | 0.0000 | 0.0000 |  | 1.0000 |

验证集方向准确率、降低方向召回、变化幅度比和区间覆盖率没有同时通过。OOT方向看似较高不能覆盖验证集失败，也不能覆盖幅度压缩。

## B3：Target → 实际 Phase 响应

精确结构模板：`{'T15760': {'selected_structural_template': '9922A1576030_PV27_002|31.0|ML09|oil_wet|465435d826d4528e|9b93edacf0e1b133', 'physical_batches': 275, 'template_selection': 'LARGEST_EXACT_STRUCTURAL_TEMPLATE'}, 'B00458': {'selected_structural_template': '0A497A75CABE', 'physical_batches': 186, 'template_selection': 'LARGEST_EXACT_STRUCTURAL_TEMPLATE'}}`。

| pilot | output | available_samples | oot_samples | oot_mae | p10_p90_coverage | conformal_radius | mae_threshold | passed | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| T15760 | dry_end_temp_actual | 275 | 42 | 6.1048 | 0.9762 | 8.6100 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| T15760 | post_oil_temp_actual | 275 | 42 | 3.4333 | 0.9524 | 4.8000 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| T15760 | wet_end_temp_actual | 275 | 42 | 0.1000 | 1.0000 | 0.2000 | 2.0000 | True | PASSED |
| T15760 | wet_duration_actual | 275 | 42 | 2.8287 | 0.9762 | 8.4304 | 5.0000 | True | PASSED |
| T15760 | pid_duration_actual | 275 | 42 | 4.7096 | 0.9286 | 0.0001 | 5.0000 | True | PASSED |
| T15760 | discharge_temp_actual | 275 | 42 | 3.7857 | 0.9524 | 4.7000 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| B00458 | dry_end_temp_actual | 186 | 29 | 2.1951 | 1.0000 | 5.4033 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| B00458 | post_oil_temp_actual | 186 | 29 | 2.3365 | 1.0000 | 4.9516 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| B00458 | wet_end_temp_actual | 186 | 29 | 3.4270 | 0.9655 | 5.4243 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| B00458 | wet_duration_actual | 186 | 29 | 0.2047 | 1.0000 | 0.3730 | 5.0000 | True | PASSED |
| B00458 | pid_duration_actual | 186 | 29 | 18.9116 | 1.0000 | 21.4497 | 5.0000 | False | PROCESS_RESPONSE_GATE_FAILED |
| B00458 | discharge_temp_actual | 186 | 29 | 2.0016 | 1.0000 | 3.1418 | 2.0000 | False | PROCESS_RESPONSE_GATE_FAILED |

即使部分终点指标通过，当前模型仍没有覆盖所有受动作影响的V3.7 Phase均值、积分与执行特征，因此两个试点统一为 `CONTROL_TRANSFER_UNAVAILABLE`。

## 扩展配方

| CompoundName | material_system | lab_groups | gate_status | policy_status |
| --- | --- | --- | --- | --- |
| M1-T09170---- 23 003 | Silica | 100 | INSUFFICIENT_STEP_SNAPSHOT | RECIPE_SPECIFIC_POLICY_REQUIRED |
| M1-T33025W3-- 04 005 | Silica | 41 | INSUFFICIENT_STEP_SNAPSHOT | RECIPE_SPECIFIC_POLICY_REQUIRED |
| M1-B00460B--- 04 007 | CarbonBlack | 7 | INSUFFICIENT_STEP_SNAPSHOT | RECIPE_SPECIFIC_POLICY_REQUIRED |
| M1-B15563R1-- 01 005 | CarbonBlack | 7 | INSUFFICIENT_STEP_SNAPSHOT | RECIPE_SPECIFIC_POLICY_REQUIRED |

四个扩展配方缺少与本合同一致的Step快照，逐个标记 `RECIPE_SPECIFIC_POLICY_REQUIRED`，不影响双试点诊断，但不能进入推荐。

## 安全边界

本实验只读取冻结快照；未访问数据库、本地 MMS 或 Archive。Stage3、配方写入、PLC 和自动控制均关闭。任何 B3 失败都禁止输出“预计降低 X MU”的 Target 推荐。

## 下一次允许开展的工作

1. 补齐 B00458 `OrderID+BatchNumber→PalletID` 的同源标签桥，但不重新读取本地Archive。
2. 对 T15760/B00458 建立完整的 Step Target→Phase均值/积分/终点响应，而不仅是终点模型。
3. 只有 Gate 0 出现 `CAUSAL_ACTION_SUPPORTED` 后才创建 DML 实验；否则继续保持方案A+B2诊断。
