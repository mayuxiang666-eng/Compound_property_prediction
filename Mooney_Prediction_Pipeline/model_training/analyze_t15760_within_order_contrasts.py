"""Natural-control analysis of large T15760 Mooney changes within one order.

The analysis is descriptive and leakage-safe: it does not train or release a
model.  Each OrderID+PalletID Lab result is counted once.  Process features are
the mean of the physical batches belonging to that exact Lab group.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
CURVE_SOURCE = BASE / "v371_r10b_phase_curve_state" / "r10b_t15760_batch_phase_curve_features.csv"
MATERIAL_SOURCE = BASE / "batch_materials_active.csv"
OUTPUT = BASE / "t15760_within_order_contrast"
COMPOUND = "M1-T15760---- 06 002"


CORE_FEATURES = [
    "Stage2_DryMixing_Duration", "Stage2_DryMixing_temp_Mean",
    "Stage2_DryMixing_temp_Std", "Stage2_DryMixing_power_Mean",
    "Stage2_DryMixing_power_Integral", "Stage2_DryMixing_Torque_Mean",
    "Stage2_DryMixing_Torque_Integral", "Stage2_DryMixing_RotorSpeed_Mean",
    "Stage2_DryMixing_Specific_Energy", "Stage2_eta_torque_End",
    "Stage3_OilLoading_Duration", "Stage3_OilLoading_temp_Mean",
    "Stage3_OilLoading_power_Integral", "Stage3_OilLoading_Torque_Integral",
    "Stage4_WetMixing_Duration", "Stage4_WetMixing_temp_Mean",
    "Stage4_WetMixing_temp_Std", "Stage4_WetMixing_power_Mean",
    "Stage4_WetMixing_power_Integral", "Stage4_WetMixing_Torque_Mean",
    "Stage4_WetMixing_Torque_Integral", "Stage4_WetMixing_RotorSpeed_Mean",
    "Stage4_WetMixing_Specific_Energy", "phys_eta_app_wetmix_mean",
    "Stage5_PID_Duration", "Stage5_PID_temp_Mean", "Stage5_PID_temp_Std",
    "Stage5_PID_power_Mean", "Stage5_PID_power_Integral",
    "Stage5_PID_Torque_Mean", "Stage5_PID_Torque_Integral",
    "Stage5_PID_RotorSpeed_Mean", "Stage5_PID_Specific_Energy",
    "pid_silanization_exposure_proxy", "pid_high_temperature_risk_proxy",
    "pid_control_instability_proxy", "pid_mechanical_work_proxy",
    "Stage6_BottomMixing_Duration", "Stage6_BottomMixing_temp_Mean",
    "Stage6_BottomMixing_temp_Std", "Stage6_BottomMixing_power_Mean",
    "Stage6_BottomMixing_power_Integral", "Stage6_BottomMixing_Specific_Energy",
    "phys_discharge_temp", "phys_max_temp", "phys_temp_integral_above_100",
    "phys_shear_history_total", "supplier_rubber_viscosity_avg",
    "supplier_silica_moisture_avg", "supplier_silica_surface_area_avg",
    "env_temp_mean", "env_humidity_mean", "BatchWeight", "Target_Temperature",
]


def normalize_id(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def phase_family(feature: str) -> str:
    if "Stage2" in feature or "DryMixing" in feature:
        return "DRY"
    if "Stage3" in feature or "OilLoading" in feature:
        return "POST_OIL"
    if "Stage4" in feature or "WetMixing" in feature or "wetmix" in feature:
        return "WET"
    if "Stage5" in feature or "pid_" in feature or "PID" in feature:
        return "PID_TOP"
    if "Stage6" in feature or "BottomMixing" in feature:
        return "PID_BOTTOM"
    if "discharge" in feature or "max_temp" in feature or "temp_integral" in feature:
        return "THERMAL_GLOBAL"
    if "supplier" in feature:
        return "RAW_MATERIAL_QM"
    if "env_" in feature:
        return "ENVIRONMENT"
    return "OTHER"


def safe_hash(values) -> str:
    payload = "|".join(sorted(set(str(value) for value in values if pd.notna(value))))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16] if payload else ""


def within_order_association(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    order_counts = frame.groupby("OrderID")["LabelGroupID"].transform("size")
    eligible = frame.loc[order_counts >= 2].copy()
    y = eligible["MNY"] - eligible.groupby("OrderID")["MNY"].transform("mean")
    for feature in features:
        values = pd.to_numeric(eligible[feature], errors="coerce")
        x = values - values.groupby(eligible["OrderID"]).transform("mean")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 30 or x.loc[valid].nunique() < 3:
            continue
        rows.append(
            {
                "feature": feature,
                "phase_family": phase_family(feature),
                "group_rows": int(valid.sum()),
                "orders": int(eligible.loc[valid, "OrderID"].nunique()),
                "within_order_pearson": float(x.loc[valid].corr(y.loc[valid], method="pearson")),
                "within_order_spearman": float(x.loc[valid].corr(y.loc[valid], method="spearman")),
                "feature_within_order_std": float(x.loc[valid].std(ddof=1)),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["abs_spearman"] = result["within_order_spearman"].abs()
        result = result.sort_values(["abs_spearman", "feature"], ascending=[False, True])
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    group = pd.read_csv(GROUP_SOURCE, low_memory=False)
    group = group.loc[group["CompoundName"].astype(str).eq(COMPOUND)].copy()
    group["OrderID"] = normalize_id(group["OrderID"])
    group["PalletID"] = normalize_id(group["PalletID"])
    group["MNY"] = pd.to_numeric(group["MNY"], errors="coerce")
    group["lab_time"] = pd.to_datetime(group["lab_time"], errors="coerce")

    phase_columns = [
        "OrderID", "BatchNumber", "LabelGroupID", "BatchStartTime", "ProcessVersion",
        "TemplateKey", "TargetVectorHash", "StepPatternHash",
    ]
    phase = pd.read_csv(PHASE_SOURCE, usecols=lambda c: c in phase_columns, low_memory=False)
    phase["OrderID"] = normalize_id(phase["OrderID"])
    phase["BatchNumber"] = normalize_id(phase["BatchNumber"])
    timing = phase.groupby("LabelGroupID", as_index=False).agg(
        production_start=("BatchStartTime", "min"),
        production_batch_count=("BatchNumber", "nunique"),
        process_version_count=("ProcessVersion", "nunique"),
        template_count=("TemplateKey", "nunique"),
        target_vector_count=("TargetVectorHash", "nunique"),
        step_pattern_count=("StepPatternHash", "nunique"),
    )
    timing["production_start"] = pd.to_datetime(timing["production_start"], errors="coerce")
    group = group.merge(timing, on="LabelGroupID", how="left", validate="one_to_one")
    group = group.sort_values(["OrderID", "production_start", "PalletID"], kind="stable")
    group["pallet_sequence"] = group.groupby("OrderID").cumcount() + 1

    curve = pd.read_csv(CURVE_SOURCE, low_memory=False)
    curve["OrderID"] = normalize_id(curve["OrderID"])
    curve["BatchNumber"] = normalize_id(curve["BatchNumber"])
    curve = phase[["OrderID", "BatchNumber", "LabelGroupID"]].merge(
        curve, on=["OrderID", "BatchNumber"], how="inner", validate="one_to_one"
    )
    curve_features = [
        column for column in curve.columns
        if column.startswith("phase_curve__")
        and column.endswith(("__last", "__mean", "__std", "__slope", "__integral", "__duration"))
    ]
    curve_group = curve.groupby("LabelGroupID", as_index=False)[curve_features].mean(numeric_only=True)
    group = group.merge(curve_group, on="LabelGroupID", how="left", validate="one_to_one")

    available_core = [feature for feature in CORE_FEATURES if feature in group.columns]
    for feature in available_core:
        group[feature] = pd.to_numeric(group[feature], errors="coerce")
    analysis_features = available_core + curve_features

    order_summary = group.groupby("OrderID", as_index=False).agg(
        group_count=("LabelGroupID", "nunique"),
        physical_batch_count=("production_batch_count", "sum"),
        mny_min=("MNY", "min"), mny_max=("MNY", "max"),
        mny_mean=("MNY", "mean"), mny_std=("MNY", "std"),
        production_start=("production_start", "min"), production_end=("production_start", "max"),
        process_versions=("ProcessVersion", "nunique"), templates=("TemplateKey", "nunique"),
    )
    order_summary["mny_range"] = order_summary["mny_max"] - order_summary["mny_min"]
    order_summary = order_summary.sort_values(["mny_range", "group_count"], ascending=[False, False])
    order_summary.to_csv(OUTPUT / "order_mny_range.csv", index=False)

    eligible_orders = order_summary.loc[order_summary["group_count"].ge(3)]
    threshold = max(8.0, float(eligible_orders["mny_range"].quantile(0.75)))
    selected_orders = eligible_orders.loc[eligible_orders["mny_range"].ge(threshold), "OrderID"].head(25).tolist()
    selected = group.loc[group["OrderID"].isin(selected_orders)].copy()
    profile_columns = [
        "OrderID", "LabelGroupID", "PalletID", "pallet_sequence", "production_start",
        "lab_time", "MNY", "ProcessVersion", "TemplateKey", "production_batch_count",
        "process_version_count", "template_count", "target_vector_count", "step_pattern_count",
    ] + available_core
    selected[profile_columns].to_csv(OUTPUT / "selected_high_variation_group_profiles.csv", index=False)

    pair_rows, contrast_rows = [], []
    robust_scale = group[analysis_features].apply(pd.to_numeric, errors="coerce").quantile(0.75) - group[analysis_features].apply(pd.to_numeric, errors="coerce").quantile(0.25)
    robust_scale = robust_scale.replace(0, np.nan)
    for order_id, order in selected.groupby("OrderID", sort=False):
        order = order.sort_values(["production_start", "PalletID"], kind="stable")
        for previous, current in zip(order.iloc[:-1].itertuples(), order.iloc[1:].itertuples()):
            row = {
                "OrderID": order_id,
                "from_group": previous.LabelGroupID,
                "to_group": current.LabelGroupID,
                "from_sequence": previous.pallet_sequence,
                "to_sequence": current.pallet_sequence,
                "delta_mny": current.MNY - previous.MNY,
            }
            for feature in available_core:
                row[f"delta__{feature}"] = getattr(current, feature) - getattr(previous, feature)
            pair_rows.append(row)
        low = order.loc[order["MNY"].idxmin()]
        high = order.loc[order["MNY"].idxmax()]
        for feature in analysis_features:
            low_value = pd.to_numeric(pd.Series([low.get(feature)]), errors="coerce").iloc[0]
            high_value = pd.to_numeric(pd.Series([high.get(feature)]), errors="coerce").iloc[0]
            delta = high_value - low_value if pd.notna(low_value) and pd.notna(high_value) else np.nan
            contrast_rows.append(
                {
                    "OrderID": order_id,
                    "low_group": low["LabelGroupID"], "high_group": high["LabelGroupID"],
                    "low_mny": low["MNY"], "high_mny": high["MNY"],
                    "delta_mny": high["MNY"] - low["MNY"],
                    "feature": feature, "phase_family": phase_family(feature),
                    "low_value": low_value, "high_value": high_value, "delta": delta,
                    "robust_scaled_abs_delta": abs(delta) / robust_scale.get(feature, np.nan) if pd.notna(delta) else np.nan,
                }
            )
    pd.DataFrame(pair_rows).to_csv(OUTPUT / "consecutive_group_deltas.csv", index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(OUTPUT / "high_vs_low_feature_contrasts.csv", index=False)
    top_contrasts = contrasts.sort_values(["OrderID", "robust_scaled_abs_delta"], ascending=[True, False]).groupby("OrderID").head(20)
    top_contrasts.to_csv(OUTPUT / "high_vs_low_top20_changes_per_order.csv", index=False)

    associations = within_order_association(group, analysis_features)
    associations.to_csv(OUTPUT / "within_order_feature_associations.csv", index=False)
    phase_association = associations.groupby("phase_family", as_index=False).agg(
        feature_count=("feature", "size"),
        median_abs_spearman=("abs_spearman", "median"),
        max_abs_spearman=("abs_spearman", "max"),
    ).sort_values("max_abs_spearman", ascending=False)
    phase_association.to_csv(OUTPUT / "phase_family_association_summary.csv", index=False)

    selected_keys = phase.loc[phase["OrderID"].isin(selected_orders), ["OrderID", "BatchNumber", "LabelGroupID"]]
    material_parts = []
    selected_set = set(selected_orders)
    for chunk in pd.read_csv(MATERIAL_SOURCE, chunksize=250_000, low_memory=False):
        chunk["OrderID"] = normalize_id(chunk["OrderID"])
        hit = chunk.loc[chunk["OrderID"].isin(selected_set)].copy()
        if len(hit):
            hit["BatchNumber"] = normalize_id(hit["BatchNumber"])
            material_parts.append(hit)
    materials = pd.concat(material_parts, ignore_index=True) if material_parts else pd.DataFrame()
    material_changes = []
    if len(materials):
        materials = materials.merge(selected_keys, on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
        material_group = materials.groupby(["OrderID", "LabelGroupID", "MaterialCode", "MaterialName"], dropna=False).agg(
            charge_signature=("ChargeID", safe_hash),
            charges=("ChargeID", lambda x: "|".join(sorted(set(str(value) for value in x if pd.notna(value))))),
            actual_weight=("ActualWeight", "sum"),
        ).reset_index()
        material_group.to_csv(OUTPUT / "selected_group_material_charge_profiles.csv", index=False)
        for order_id, order in selected.groupby("OrderID", sort=False):
            low = order.loc[order["MNY"].idxmin()]
            high = order.loc[order["MNY"].idxmax()]
            low_material = material_group.loc[material_group["LabelGroupID"].eq(low["LabelGroupID"])]
            high_material = material_group.loc[material_group["LabelGroupID"].eq(high["LabelGroupID"])]
            comparison = low_material.merge(
                high_material, on=["OrderID", "MaterialCode", "MaterialName"], how="outer", suffixes=("_low", "_high")
            )
            comparison["charge_changed"] = comparison["charge_signature_low"].fillna("") != comparison["charge_signature_high"].fillna("")
            for _, item in comparison.loc[comparison["charge_changed"]].iterrows():
                material_changes.append(
                    {
                        "OrderID": order_id, "low_group": low["LabelGroupID"], "high_group": high["LabelGroupID"],
                        "delta_mny": high["MNY"] - low["MNY"], "MaterialCode": item["MaterialCode"],
                        "MaterialName": item["MaterialName"], "charges_low": item.get("charges_low", ""),
                        "charges_high": item.get("charges_high", ""), "weight_low": item.get("actual_weight_low", np.nan),
                        "weight_high": item.get("actual_weight_high", np.nan),
                    }
                )
    pd.DataFrame(material_changes).to_csv(OUTPUT / "high_vs_low_material_charge_changes.csv", index=False)

    top_assoc = associations.head(20).to_dict("records") if len(associations) else []
    phase_counts = top_contrasts.groupby("phase_family").size().sort_values(ascending=False).to_dict()
    summary = {
        "contract": "T15760-WITHIN-ORDER-NATURAL-CONTRAST-V1",
        "compound": COMPOUND,
        "groups": int(len(group)),
        "orders": int(group["OrderID"].nunique()),
        "orders_with_at_least_2_groups": int((order_summary["group_count"] >= 2).sum()),
        "orders_with_at_least_3_groups": int((order_summary["group_count"] >= 3).sum()),
        "selection_range_threshold_mny": threshold,
        "selected_high_variation_orders": len(selected_orders),
        "selected_order_ids": selected_orders,
        "selected_range_median": float(order_summary.loc[order_summary["OrderID"].isin(selected_orders), "mny_range"].median()),
        "selected_range_max": float(order_summary.loc[order_summary["OrderID"].isin(selected_orders), "mny_range"].max()),
        "top_within_order_associations": top_assoc,
        "top20_contrast_feature_family_counts": phase_counts,
        "interpretation_guard": "ASSOCIATION_NOT_CAUSATION; Lab label is one value per OrderID+PalletID group",
        "local_mms_access": False,
        "database_access_during_run": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
