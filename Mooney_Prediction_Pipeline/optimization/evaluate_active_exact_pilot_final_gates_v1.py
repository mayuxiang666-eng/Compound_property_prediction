"""Consolidate authoritative Active-only gates for the two exact pilots."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "corrected_phase_history"
STEP = REPORT / "active_exact_pilot_step_history_v1"
OUTPUT = STEP / "active_exact_pilot_final_gate_matrix_v1.csv"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    exact = load(REPORT / "v371_active_exact_pilot_backtests_v1" / "exact_pilot_backtest_summary.json")["candidates"]
    templates = load(STEP / "active_exact_pilot_template_contract_summary_v2.json")["recipes"]
    actions = load(STEP / "active_template_action_support_summary_v1.json")["templates"]
    process = load(STEP / "b00458_main_template_process_response_v1" / "summary.json")
    latest = load(STEP / "exact_pilot_latest_active_orders_summary.json")["compounds"]

    definitions = {
        "b00458_12004": {
            "compound": "M1-B00458---- 12 004", "recipe": "9922A00458BT_PV77_004",
            "template": "0A497A75CABE", "model_key": "b00458_12004",
        },
        "t09170ws_02004": {
            "compound": "M1-T09170WS-- 02 004", "recipe": "9922A09170AX_PV50_004",
            "template": "BF3EC40AB5D1", "model_key": "t09170ws_02004",
        },
    }
    rows = []
    for pilot, definition in definitions.items():
        recipe, template = definition["recipe"], definition["template"]
        model = exact[definition["model_key"]]
        template_data = templates[recipe]
        action_data = actions[f"{recipe}|{template}"]
        latest_time = pd.Timestamp(latest[definition["compound"]]["latest_order_time"])
        checks = [
            ("BASELINE_COMPLETE_LABEL_GROUPS_MIN_150", model["exact_group_counts"]["total"] >= 150,
             model["exact_group_counts"]["total"]),
            ("BASELINE_OOT_EXISTS", model["exact_oot_metrics"]["groups"] >= 20,
             model["exact_oot_metrics"]["groups"]),
            ("RECIPE_TEMPLATE_CONSISTENCY_90PCT", template_data["template_count"] == 1 and template_data["templates"][0]["share"] >= 0.9,
             max(item["share"] for item in template_data["templates"])),
            ("SELECTED_TEMPLATE_PROCESS_SAMPLES_MIN_150", next(item["batches"] for item in template_data["templates"] if item["template_id"] == template) >= 150,
             next(item["batches"] for item in template_data["templates"] if item["template_id"] == template)),
            ("SCP_CONTRACT_COMPLETE", template_data["contract_coverage_pass"], template_data["missing_contract_steps"]),
            ("IDENTIFIABLE_ACTION_MIN_3_LEVELS", action_data["pass_min_3_levels"] > 0, action_data["pass_min_3_levels"]),
            ("DISCHARGE_TRACKING_PASS", action_data["discharge_tracking_pass"], action_data["discharge_tracking_pass"]),
            ("CURRENT_ACTIVE_ORDER_AFTER_2026_01_01", latest_time >= pd.Timestamp("2026-01-01"), latest_time.isoformat()),
            ("ENDPOINT_RESPONSE_APPROVED", process["endpoint_response_approved"] if pilot == "b00458_12004" else False,
             process["endpoint_response_approved"] if pilot == "b00458_12004" else "NOT_IDENTIFIABLE"),
            ("ACTUAL_V37_PHASE_FEATURE_RESPONSE_APPROVED", False, "NOT_AVAILABLE"),
            ("THERMAL_BUDGET_MODEL_PASS", False, "NOT_AVAILABLE"),
            ("GROUP_MOONEY_RESPONSE_GE_0_25", False, "NOT_EVALUATED_CONTROL_TRANSFER_BLOCKED"),
        ]
        for gate, passed, evidence in checks:
            rows.append({
                "Pilot": pilot, "CompoundName": definition["compound"], "RecipeID": recipe,
                "TemplateID": template, "Gate": gate, "Passed": bool(passed), "Evidence": evidence,
            })
    matrix = pd.DataFrame(rows)
    matrix.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    result = {
        "contract": "ACTIVE-EXACT-PILOT-FINAL-GATES-V1",
        "pilots": {
            pilot: {
                "passed_gates": int(frame["Passed"].sum()),
                "total_gates": int(len(frame)),
                "all_gates_pass": bool(frame["Passed"].all()),
                "manual_review_vector_status": "ELIGIBLE" if frame["Passed"].all() else "BLOCKED",
                "failed_gates": frame.loc[~frame["Passed"], "Gate"].tolist(),
            }
            for pilot, frame in matrix.groupby("Pilot")
        },
        "production_release": False, "database_writes": False,
        "automatic_control": "DISABLED", "recommendations_generated": 0,
    }
    (STEP / "active_exact_pilot_final_gate_summary_v1.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
