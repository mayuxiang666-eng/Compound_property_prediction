# 2026-08-14 M1 半监督 Teacher–Student 实验

## Scope and safety

- Strict scope: M1 only.
- Data source: existing local snapshots only; no MMS/Archive access during this experiment.
- Stage3 online calibration: disabled.
- Release: development experiment only; no recommendation vector generated.

## Method

- Labeled input: `reports/v371_challenger/global_rds_active_v1/m1_temporal_jit_v1/m1_temporal_jit_acceptance_input.csv`.
- Unlabeled input: `reports/v371_challenger/global_rds_active_v1/v371_r1b_m1_only_snapshot_v1/m1_only_batch_pid_quality_features.csv`.
- 1,173 labeled groups and 24 groups without Lab were found.
- Only 15 unlabeled groups were before the locked OOT start and entered the pseudo-label pool.
- Teacher: five-seed ExtraTrees ensemble on the labeled train split.
- Pseudo-label gates: ensemble standard deviation <= 1.5 MU and nearest train feature distance <= 3.5 standardized units.
- Student: same model family, real labels weight 1.0, accepted pseudo-label weight 0.15.
- OOT scoring uses real Lab groups only and is not changed by pseudo-labels.

## Result

| Model | OOT MAE | OOT RMSE | Bias | R2 | Pred/Actual Std | Direction | Delta Pearson | Delta Std Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 4.556 | 7.005 | -0.253 | 0.715 | 0.834 | 44.8% | -0.026 | 0.186 |
| Student | 4.568 | 6.997 | -0.115 | 0.716 | 0.841 | 55.2% | -0.014 | 0.181 |

Interpretation: pseudo-labeling slightly improved R2 and direction accuracy, but did not improve MAE and did not recover order-to-order amplitude. It therefore does not meet the 2.5 MU / trend-capture target and is not promoted.

## Artifacts

- `reports/v371_challenger/global_rds_active_v1/m1_semi_supervised_teacher_student_v1/m1_semi_supervised_metrics.csv`
- `reports/v371_challenger/global_rds_active_v1/m1_semi_supervised_teacher_student_v1/m1_semi_supervised_oot_predictions.csv`
- `reports/v371_challenger/global_rds_active_v1/m1_semi_supervised_teacher_student_v1/m1_semi_supervised_pseudo_pool_audit.csv`
- `reports/v371_challenger/global_rds_active_v1/m1_semi_supervised_teacher_student_v1/m1_semi_supervised_actual_vs_predicted_fullview.png`
- `reports/v371_challenger/global_rds_active_v1/m1_semi_supervised_teacher_student_v1/M1半监督TeacherStudent实验报告.md`

## Decision

Keep the semi-supervised framework as an experiment, but do not use pseudo-labels to mask the main problem. The next improvement must target process-state representation and within-order amplitude; expanding the unlabeled pool later may help domain adaptation but cannot by itself guarantee the 2.5 MU target.
