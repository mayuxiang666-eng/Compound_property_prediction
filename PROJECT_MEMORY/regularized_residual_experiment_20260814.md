# 2026-08-14 M1 正则化过程残差实验记忆

## Implementation

Added `Mooney_Prediction_Pipeline/model_training/run_m1_regularized_transition_residual_v1.py`.

The model uses a time-safe expanding history of same-order M1 transitions. Features include Dry/Wet/PID/bottom-mixing/discharge process deltas, current process state, transition gap, PID risk indicators, and curve statistical risk. Missing history is blocked; current/future Lab values are excluded. The model is a pooled Ridge with robust scaling and median imputation.

## Result

- OOT transition count: 43; regularized model valid predictions: 42.
- Carry-forward previous Lab MNY: MAE 4.510, Std ratio 0.979.
- Regularized process residual: MAE 7.723, RMSE 21.605, Bias 2.899, Std ratio 2.146.
- T15760 OOT: 21 transitions, MAE 5.665, Std ratio 1.039, significant direction accuracy 0.333.

## Decision

Reject this model. The process-difference mapping is unstable and overreacts with the current sparse transition history. Do not tune alpha against OOT or use it for recommendations. Preserve the plot and audit outputs as a negative control.

## Next

Increase strict same-order transition coverage and build a hierarchical residual contract with explicit Recipe/Template OOD, material-lot stratification, detection interval, curve-risk flags, and minimum transition counts. Keep Stage3 disabled and the recommendation chain closed.
