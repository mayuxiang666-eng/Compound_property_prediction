# 2026-08-14 M1 同订单前序状态消融记忆

## Scope and safety

- Scope is strictly M1-only. M2/M3/PB/R1 are excluded from training, validation, acceptance, and recommendation.
- Data is from the cloud RDS APAC Active snapshot artifacts only.
- Local MMS and local Archive were not accessed. Cloud Archive remains `BLOCKED_ARCHIVE_NOT_REPLICATED`.
- Stage3 calibration remains disabled; no Recipe/PLC writes; energy is audit-only.

## Change

Added `Mooney_Prediction_Pipeline/model_training/prepare_v371_r1b_m1_order_state_contract_v1.py`.

The contract adds causal same-order prior process state. A prior state is eligible only when it was completed before the current batch started and the prior/current `TemplateKey` agrees. No Lab value, future batch, or cross-template numeric state is used.

Artifacts:

- `reports/v371_challenger/global_rds_active_v1/v371_r1b_m1_order_state_snapshot_v1/`
- `reports/v371_challenger/global_rds_active_v1/v371_r1b_m1_order_state/`
- `reports/v371_challenger/global_rds_active_v1/v371_r1b_m1_order_state_acceptance/`
- `docs/M1_order_state_ablation_report_20260814.md`

## Evidence

- 4,100 M1 batches and 1,197 M1 Lab groups.
- Causal prior-state availability: 19.3818%.
- Same-template rate among available prior states: 100%.
- Observed template/process-version/step-pattern change counts: 0.
- Strict acceptance compares per-batch predictions averaged within the matching `OrderID|PalletID` Lab group against one Lab MNY value. It does not repeat a Lab value per batch.

## Result and decision

The state-augmented R1B model is diagnostic only and is not promoted:

- M1-T15760 OOT MAE: 5.122 baseline -> 5.197 state model.
- M1-T15760 prediction Std / actual Std: 0.514 -> 0.516.
- M1-T15760 significant direction accuracy: about 0.438 baseline -> 0.313 state model.
- M1-B00458 OOT MAE improved slightly, but prediction amplitude remains far below the acceptance range and the added state layer is sparse.

The strict M1-only R1B baseline remains the reference challenger. The order-state contract is retained as an auditable diagnostic layer, not as a released feature set.

## Next action

Build the M1 temporal JIT/local probabilistic expert. Neighbors and calibration candidates must use only data available before the scored batch; the expert must return prediction interval and OOD status, and must fall back by explicit M1 route/material hierarchy rather than selecting an arbitrary expert. Only after this passes T15760 plus one M1 carbon-black trial should Step Target -> actual Phase gray-box response and constrained offline optimization resume.
