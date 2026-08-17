"""R11 V2 runner with strict within-OrderID trend selection and metrics."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from Mooney_Prediction_Pipeline.model_training import run_v371_r11_step_gated_delta as r11
from Mooney_Prediction_Pipeline.model_training.order_atomic_trend_metrics import candidate_score, metric_record


ROOT = Path(__file__).resolve().parents[2]
r11.OUTPUT = r11.BASE / "v371_r11_step_gated_delta_v2"
r11.MODEL_OUTPUT = ROOT / "Mooney_Prediction_Pipeline" / "models" / "v371_challenger_r11_step_gated_delta_v2"
r11.candidate_score = candidate_score
r11.metric_record = metric_record


def main():
    r11.main()
    summary_path = r11.OUTPUT / "r11_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["contract"] = "V3.7.1-R11-STEP-BOUNDARY-GATED-ORDER-DELTA-V2"
    summary["trend_evaluation"] = "CONSECUTIVE_LABEL_GROUPS_WITHIN_SAME_ORDERID_ONLY"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    model_path = r11.MODEL_OUTPUT / "r11_step_gated_delta_shadow.joblib"
    bundle = joblib.load(model_path)
    bundle["contract"] = summary["contract"]
    bundle["trend_evaluation"] = summary["trend_evaluation"]
    joblib.dump(bundle, model_path)
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
