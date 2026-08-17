# 2026-08-14 M1 分层 OOD 合同记忆

## Implementation

Added `Mooney_Prediction_Pipeline/model_training/build_m1_hierarchical_ood_residual_contract_v1.py`.

The contract uses only earlier same-order transition history and applies explicit minimum counts: exact Recipe/Template/Mixer/Route 15, Recipe 20, Material/Route/Mixer 30, Material System 80. It distinguishes missing prior Lab, sparse transition history, eligible history scope, and model promotion gate.

## Evidence

- M1 groups: 1,197.
- Strict same-order transitions: 239.
- OOD-eligible history scope: 153.
- Recommendation gate open: 0, because the candidate models have not passed model acceptance.
- OOT scope counts: no prior Lab 168, sparse 10, Recipe 22, Material/Route/Mixer 4, Material System 7.

## Decision

Do not expand transition samples by reusing a Lab value, mixing different orders, or reading local MMS/Archive. The contract is an engineering gate only; it does not imply model readiness. The recommendation chain remains closed.

## Next

Obtain more strictly aligned M1 Active Lab transitions over time, and then fit/evaluate a hierarchical residual expert with Recipe/Template OOD, material-lot, detection interval, curve-risk, and minimum-history controls.
