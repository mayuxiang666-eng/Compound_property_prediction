import pandas as pd

from Mooney_Prediction_Pipeline.model_training.run_v371_c1_history_weighted_h2_v1 import (
    assign_history_weights,
    build_history_eligibility,
)


def _batch(groups, recipe="R1", mixer="ML01", route="oil_wet", temperature=165.0):
    rows = []
    for group in groups:
        rows.append({
            "LabelGroupID": group,
            "RecipeID": recipe,
            "MixerLine": mixer,
            "phase_route": route,
            "Target_Temperature": temperature,
            "Top_Fill_Factor": 60.0,
            "Bot_Fill_Factor": 0.0,
            "weight_pct_solid_elastomer": 30.0,
            "weight_pct_natural_rubber": 5.0,
            "weight_pct_silica": 35.0,
            "weight_pct_oil": 10.0,
            "weight_pct_silian": 3.0,
            "weight_pct_carbon_black": 2.0,
        })
    return pd.DataFrame(rows)


def test_static_contract_blocks_temperature_mismatch():
    current_ids = [f"C{i}" for i in range(10)]
    current = _batch(current_ids)
    history = pd.concat([_batch(["H1"]), _batch(["H2"], temperature=160.0)], ignore_index=True)
    current_groups = pd.DataFrame({"LabelGroupID": current_ids, "split": "train"})
    history_groups = pd.DataFrame({
        "LabelGroupID": ["H1", "H2"],
        "lab_time": ["2025-01-01", "2025-01-01"],
        "FixedValidationOOTStart": ["2025-07-01", "2025-07-01"],
        "temporal_cutoff_status": ["SAFE", "SAFE"],
    })
    audit = build_history_eligibility(current, history, current_groups, history_groups)
    status = audit.set_index("LabelGroupID")["eligibility_status"]
    assert status["H1"] == "ELIGIBLE_STATIC_PROCESS_CONTRACT_MATCH"
    assert status["H2"] == "BLOCKED_STATIC_PROCESS_CONTRACT_MISMATCH"


def test_recipe_weight_cap_is_enforced():
    current_ids = [f"C{i}" for i in range(10)]
    current = _batch(current_ids)
    current_groups = pd.DataFrame({"LabelGroupID": current_ids, "split": "train"})
    eligibility = _batch(["H1", "H2", "H3", "H4"])[
        ["LabelGroupID", "RecipeID"]
    ].copy()
    eligibility["eligibility_status"] = "ELIGIBLE_STATIC_PROCESS_CONTRACT_MATCH"
    eligibility["lab_time"] = "2025-06-30"
    eligibility["FixedValidationOOTStart"] = "2025-07-01"
    weighted = assign_history_weights(
        eligibility, current, current_groups, base_weight=1.0, half_life_days=180.0,
        recipe_cap_fraction=0.25,
    )
    assert weighted["_group_training_weight"].sum() <= 2.5 + 1e-12
    assert (weighted["_group_training_weight"] > 0).all()
