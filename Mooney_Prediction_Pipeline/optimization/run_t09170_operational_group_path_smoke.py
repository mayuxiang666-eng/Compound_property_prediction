"""Smoke-test the repaired no-Lab group path on a real T09170 order."""

from __future__ import annotations

import json
from pathlib import Path

from Mooney_Prediction_Pipeline.optimization.v37_group_feature_path import build_v37_group_feature_path
from Mooney_Prediction_Pipeline.optimization.v37_raw_predictor import V37RawPredictor


ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v37_production_model_package"
OUT = ROOT / "Mooney_Prediction_Pipeline" / "reports" / "expert_step_vector_shadow"
ORDER_ID = "2016836"


def main() -> None:
    predictor = V37RawPredictor(PACKAGE).load()
    result = build_v37_group_feature_path(ORDER_ID, predictor)
    OUT.mkdir(parents=True, exist_ok=True)
    batch_path = OUT / "t09170ws_02004_operational_group_path_batch_predictions.csv"
    group_path = OUT / "t09170ws_02004_operational_group_path_pallet_predictions.csv"
    audit_path = OUT / "t09170ws_02004_operational_group_path_summary.json"
    result.batch_predictions.to_csv(batch_path, index=False, encoding="utf-8-sig")
    result.group_predictions.to_csv(group_path, index=False, encoding="utf-8-sig")
    audit_path.write_text(json.dumps(result.audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result.audit, ensure_ascii=False, indent=2))
    print(result.group_predictions.to_string(index=False))
    print(f"batch={batch_path}")
    print(f"group={group_path}")
    print(f"summary={audit_path}")


if __name__ == "__main__":
    main()
