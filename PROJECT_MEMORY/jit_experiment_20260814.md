# 2026-08-14 M1 时间安全 JIT 局部专家记忆

## Implementation

Added `Mooney_Prediction_Pipeline/model_training/run_m1_temporal_jit_expert_v1.py`.

The diagnostic predicts each Lab group using only strictly earlier Lab groups. Scope hierarchy is exact Recipe/Template/Mixer/Route, Recipe, Material/Route/Mixer, then Material System. It uses robust historical IQR distance over 22 interpretable static/process features and returns a weighted point estimate, P10/P90 interval, neighbor count, scope, nearest distance, and OOD status.

## Results

- M1 OOT: 211 groups, valid point prediction coverage 100%; `IN_DOMAIN` rate about 72.0%.
- M1-T15760 OOT MAE 4.493, prediction Std/actual Std 0.334, significant direction accuracy 0.375.
- M1-B00458 OOT MAE 1.480, prediction Std/actual Std 0.173; direction evidence is only one pair.
- M1-T09170 OOT MAE 1.626, prediction Std/actual Std 0.510; direction evidence is only one pair.
- M1-T33025 OOT MAE 5.066, prediction Std/actual Std 0.127.

## Decision

JIT demonstrates useful local level information, especially for T15760, but it does not meet the joint MAE and volatility requirement. It is not a released model and cannot generate recommendations. OOD rows remain non-recommendable.

## Next

Build a time-safe same-order `Delta MNY` expert and a process-sensitive residual expert. Do not use current/future Lab values, do not open Stage3, and do not resume Step Target recommendation until T15760 and one M1 carbon-black trial pass together.
