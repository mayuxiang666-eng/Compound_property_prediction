# 2026-08-14 M1 半监督过程表征 V2

## Method

- No MMS/Archive/database access.
- PCA representation fit only on batches before the locked OOT start (`2025-12-08 07:15:36 UTC`), preventing future OOT feature distribution from fitting the representation.
- 128 pre-OOT batch rows, 156 common numeric process features, 24 PCA components.
- For each LabelGroup, latent process-state means and standard deviations were added to the supervised model.
- 15 pre-OOT unlabeled groups passed teacher ensemble agreement and feature-distance gates; pseudo-label weight remained 0.15.

## OOT result

| Model | MAE | RMSE | Bias | R2 | Pred/Actual Std | Direction | Delta Pearson | Delta Std Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Representation Teacher | 4.576 | 7.032 | -0.184 | 0.713 | 0.835 | 55.2% | -0.053 | 0.174 |
| Representation Student | 4.642 | 7.089 | -0.289 | 0.708 | 0.833 | 58.6% | -0.001 | 0.233 |

## Decision

Direction accuracy improved relative to the first semi-supervised run, but MAE worsened and order-level amplitude remains far below 1.0. Do not promote. The experiment confirms that unlabeled process representation alone cannot solve the missing within-order state/label alignment problem.

## Artifacts

- `reports/v371_challenger/global_rds_active_v1/m1_semisupervised_representation_v2/m1_representation_metrics.csv`
- `reports/v371_challenger/global_rds_active_v1/m1_semisupervised_representation_v2/m1_representation_oot_predictions.csv`
- `reports/v371_challenger/global_rds_active_v1/m1_semisupervised_representation_v2/m1_representation_actual_vs_predicted_fullview.png`
- `reports/v371_challenger/global_rds_active_v1/m1_semisupervised_representation_v2/M1半监督过程表征V2实验报告.md`
