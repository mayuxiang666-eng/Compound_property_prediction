# 2026-08-14 M1 同订单 Delta MNY / 过程残差实验记忆

## Implementation

Added `Mooney_Prediction_Pipeline/model_training/run_m1_order_delta_residual_expert_v1.py`.

The transition unit is the current Lab group minus the immediately previous Lab group in the same OrderID. It requires same RecipeID and TemplateKey, strictly increasing Lab time, and an available previous MNY. The Delta JIT neighbor pool contains only earlier transitions. Dry/Wet/PID/bottom-mixing/discharge process differences are used as distance features.

## Results

- 1,197 M1 Lab groups and 958 orders produced 239 eligible transitions; OOT has 43 transitions.
- OOT carry-forward previous MNY: MAE 4.510, prediction Std/actual Std 0.979.
- OOT Delta JIT: MAE 5.330, prediction Std/actual Std 1.081.
- Fixed 50/50 level/delta blend: MAE 4.764, prediction Std/actual Std 0.999.
- T15760 OOT: 21 transitions; carry-forward MAE 5.578, Delta JIT MAE 6.451, direction accuracy 0.250.
- B00458 OOT has only one transition, so no independent conclusion is allowed.

## Decision

Delta JIT and the fixed blend are not promoted. Carry-forward is retained only as a diagnostic baseline. The experiment confirms same-order persistence but not a stable process-difference-to-Delta-MNY mapping. No Step Target recommendation is enabled.

## Next

Increase strictly aligned same-order transition coverage and add detection interval, material-lot, process risk, and sensitive-stage deltas to a regularized residual expert. Require sufficient OOT transitions for T15760 and one M1 carbon-black trial before resuming the gray-box recommendation chain.
