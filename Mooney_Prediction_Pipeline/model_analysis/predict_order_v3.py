import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_DIR = os.path.dirname(SCRIPT_DIR)
WORKSPACE_DIR = os.path.dirname(PIPELINE_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(PIPELINE_DIR, "model_training"))

from predict_latest_sfe_order import connect_mms, extract_features_for_row
from train_group_mooney_models_ultimate3stage import RobustUltimate3StageModel


# The current V3 bundles were trained by running the training script directly,
# so pickle records this class under __main__. Register it for compatible loading.
setattr(sys.modules["__main__"], "RobustUltimate3StageModel", RobustUltimate3StageModel)


TRACK_FOLDERS = {
    (True, False): "results_with_oil",
    (False, False): "results_without_oil",
    (True, True): "results_silica_with_oil",
    (False, True): "results_silica_without_oil",
}


def query_order_rows(order_id):
    sql = """
    WITH order_batches AS (
        SELECT
            a.CompoundDescription,
            a.OrderID,
            a.CompoundName,
            a.OrderStartTime,
            a.Equipment AS MixerLine,
            bh.BatchNumber,
            bh.BatchWeight,
            bc.Curve1 AS temp,
            bc.Curve2 AS power,
            bc.Curve5 AS Torque,
            bc.Curve6 AS RotorSpeed,
            bc.Curve7 AS WayofRam
        FROM dbo.Orders a
        JOIN dbo.BatchHeader bh
            ON bh.OrderID = a.OrderID
        OUTER APPLY (
            SELECT TOP 1 Curve1, Curve2, Curve5, Curve6, Curve7
            FROM dbo.BatchCurve bc
            WHERE bc.OrderID = a.OrderID
              AND bc.BatchNumber = bh.BatchNumber
            ORDER BY bc.Timestamp DESC
        ) bc
        WHERE a.OrderID = ?
          AND bc.Curve1 IS NOT NULL
          AND bc.Curve2 IS NOT NULL
    ),
    material_totals AS (
        SELECT
            om.OrderID,
            ob.BatchNumber,
            SUM(CASE WHEN om.MaterialCode LIKE 'CE%' AND om.MaterialCode NOT LIKE 'CE19%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_solid_elastomer,
            SUM(CASE WHEN om.MaterialCode LIKE 'CN%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_natural_rubber,
            SUM(CASE WHEN om.MaterialCode LIKE 'CS100%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_silica,
            SUM(CASE WHEN om.MaterialCode LIKE 'CS%' AND om.MaterialCode NOT LIKE 'CS100%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_oil,
            SUM(CASE WHEN om.MaterialCode LIKE 'CA551%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_silian,
            SUM(CASE WHEN om.MaterialCode LIKE 'CC%'
                THEN CAST(om.BatchWeight AS FLOAT) / NULLIF(CAST(ob.BatchWeight AS FLOAT), 0) * 100 END) AS weight_pct_carbon_black
        FROM dbo.OrderMaterials om
        JOIN order_batches ob
            ON ob.OrderID = om.OrderID
        GROUP BY om.OrderID, ob.BatchNumber, ob.BatchWeight
    )
    SELECT
        ob.*,
        oil.Value AS CurrentValue,
        oil.PrevStepValue,
        top_fill.ParameterValue AS Top_Fill_Factor,
        bot_fill.ParameterValue AS Bot_Fill_Factor,
        target_temp.ParameterValue AS Target_Temperature,
        mt.weight_pct_solid_elastomer,
        mt.weight_pct_natural_rubber,
        mt.weight_pct_silica,
        mt.weight_pct_oil,
        mt.weight_pct_silian,
        mt.weight_pct_carbon_black
    FROM order_batches ob
    OUTER APPLY (
        SELECT TOP 1
            bd.Value,
            bd_prev.Value AS PrevStepValue
        FROM dbo.BatchData bd
        LEFT JOIN dbo.BatchData bd_prev
            ON bd_prev.OrderID = bd.OrderID
           AND bd_prev.BatchNumber = bd.BatchNumber
           AND bd_prev.EquipmentID = bd.EquipmentID
           AND bd_prev.StepNo = bd.StepNo - 1
           AND bd_prev.VariablePath = 'SCP-1-Step-Time-rel-s'
           AND bd_prev.GroupName = 'AVR_MST'
        WHERE bd.OrderID = ob.OrderID
          AND bd.BatchNumber = ob.BatchNumber
          AND bd.VariablePath = 'SCP-1-Step-Time-rel-s'
          AND bd.GroupName = 'AVR_MST'
          AND EXISTS (
              SELECT 1
              FROM dbo.RecipeMaterials rm
              WHERE rm.RecipeID = ob.CompoundDescription
                AND rm.StepNumber = bd.StepNo
                AND rm.MaterialCode LIKE '%CS4%'
          )
    ) oil
    LEFT JOIN dbo.RecipeCBS3Parameters top_fill
        ON top_fill.RecipeID = ob.CompoundDescription
       AND top_fill.ParameterID = 'MI.1.RHT.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters bot_fill
        ON bot_fill.RecipeID = ob.CompoundDescription
       AND bot_fill.ParameterID = 'MI.1.RHB.Fill-Factor'
    LEFT JOIN dbo.RecipeCBS3Parameters target_temp
        ON target_temp.RecipeID = ob.CompoundDescription
       AND target_temp.ParameterID = 'MI.1.RHT.Target-Temperature'
    LEFT JOIN material_totals mt
        ON mt.OrderID = ob.OrderID
       AND mt.BatchNumber = ob.BatchNumber
    ORDER BY ob.BatchNumber
    """
    with connect_mms("SFEPLANT") as connection:
        return pd.read_sql(sql, connection, params=[str(order_id)])


def select_bundle(feature_row):
    is_oil = float(feature_row["is_oil_loading_present"]) == 1.0
    silica_phr = float(feature_row.get("silica_phr", 0.0))
    silane_pct = float(feature_row.get("weight_pct_silian", 0.0) or 0.0)
    is_silica = silica_phr >= 25.0 and silane_pct > 0.0
    track_folder = TRACK_FOLDERS[(is_oil, is_silica)]
    bundle_path = os.path.join(
        WORKSPACE_DIR,
        track_folder,
        "mooney_ultimate3stage_cluster_bundle.joblib",
    )
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"V3 model bundle not found: {bundle_path}")
    return track_folder, bundle_path


def predict_order(order_id):
    order_rows = query_order_rows(order_id)
    if order_rows.empty:
        raise ValueError(f"No complete temperature/power curves found for OrderID {order_id}.")

    feature_rows = []
    failures = []
    for _, order_row in order_rows.iterrows():
        try:
            feature_rows.append(extract_features_for_row(order_row))
        except Exception as error:
            failures.append({
                "OrderID": order_id,
                "BatchNumber": order_row["BatchNumber"],
                "error": str(error),
            })

    if not feature_rows:
        raise ValueError(f"No batch in OrderID {order_id} could be feature-engineered: {failures}")

    feature_df = pd.DataFrame(feature_rows)
    track_folder, bundle_path = select_bundle(feature_df.iloc[0])
    bundle = joblib.load(bundle_path)
    model = bundle["model"]

    # Missing supplier/weather measurements remain NaN and are imputed by the fitted model pipeline.
    for column in bundle["recipe_cols"] + bundle["process_cols"]:
        if column not in feature_df.columns:
            feature_df[column] = np.nan

    # A missing raw-material fraction is unknown, not zero. The shared feature
    # extractor returns zero ratios when their denominators are unavailable;
    # restore those values to missing so the fitted training imputer is used.
    missing_fraction_inputs = (
        feature_df[[
            "weight_pct_solid_elastomer",
            "weight_pct_natural_rubber",
            "weight_pct_carbon_black",
        ]]
        .isna()
        .all(axis=1)
    )
    feature_df.loc[missing_fraction_inputs, [
        "ratio_nr_rubber",
        "ratio_filler_polymer",
    ]] = np.nan

    predictions = model.predict(feature_df, apply_stage3_aakf=False)
    output = feature_df[[
        "OrderID", "BatchNumber", "CompoundName", "MixerLine", "OrderStartTime",
        "is_oil_loading_present", "silica_phr",
    ]].copy()
    output["predicted_MNY_v3_stage1_stage2"] = predictions
    output["model_track"] = bundle["track_name"]
    output["model_bundle"] = os.path.relpath(bundle_path, WORKSPACE_DIR)
    output["stage3_applied"] = False
    output["source"] = "SFEPLANT real-time order, recipe, settings, and mixing curves"
    return output, failures


def main():
    parser = argparse.ArgumentParser(description="Predict every complete batch in one SFEPLANT order using the V3 model.")
    parser.add_argument("--order-id", required=True, help="Exact SFEPLANT OrderID to predict.")
    parser.add_argument("--output", help="Optional CSV output path.")
    args = parser.parse_args()

    prediction_df, failures = predict_order(args.order_id)
    output_path = args.output or os.path.join(
        WORKSPACE_DIR,
        "scratch",
        f"order_{args.order_id}_v3_predictions.csv",
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prediction_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print(prediction_df.to_string(index=False))
    print(f"\nSaved {len(prediction_df)} batch predictions to: {output_path}")
    if failures:
        print(f"Feature extraction failures ({len(failures)}): {failures}")


if __name__ == "__main__":
    main()