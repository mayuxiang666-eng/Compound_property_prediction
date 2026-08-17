"""T15760 within-order Step execution analysis from frozen RDS Active BatchData."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
PHASE_SOURCE = BASE / "phase_batch_features_active_v1.csv"
GROUP_SOURCE = BASE / "pallet_group_training_frame_c2_prepared.csv"
BATCHDATA_DIR = BASE / "batchdata_by_order_chunks10"
OUTPUT = BASE / "t15760_within_order_contrast" / "step_execution"
COMPOUND = "M1-T15760---- 06 002"
VARIABLES = {
    "SCP-1-Step-Time-rel-s": "time",
    "SCP-2-Energy-rel-kWh": "energy",
    "SCP-3-Temperature-C-F": "temp",
    "Rotor-Speed-1/min-Rpm": "rotor",
}


def normalize(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def build_batch_step_features(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["StepNo"] = pd.to_numeric(rows["StepNo"], errors="coerce").astype("Int64")
    rows["Value"] = pd.to_numeric(rows["Value"], errors="coerce")
    rows["Target"] = pd.to_numeric(rows["Target"], errors="coerce")
    rows["parameter"] = rows["VariablePath"].map(VARIABLES)
    rows = rows.dropna(subset=["StepNo", "parameter"])
    index = ["OrderID", "BatchNumber", "LabelGroupID", "GroupName", "StepNo"]
    value = rows.pivot_table(index=index, columns="parameter", values="Value", aggfunc="last").reset_index()
    target = rows.pivot_table(index=index, columns="parameter", values="Target", aggfunc="last").reset_index()
    target = target.rename(columns={name: f"{name}_target" for name in VARIABLES.values() if name in target.columns})
    step = value.merge(target, on=index, how="outer", validate="one_to_one")
    step = step.sort_values(["OrderID", "BatchNumber", "GroupName", "StepNo"], kind="stable")
    keys = ["OrderID", "BatchNumber", "GroupName"]
    for name in ("time", "energy"):
        if name in step:
            previous = step.groupby(keys)[name].shift(1).fillna(0.0)
            step[f"{name}_increment"] = step[name] - previous
        target_name = f"{name}_target"
        if target_name in step:
            previous = step.groupby(keys)[target_name].shift(1).fillna(0.0)
            step[f"{name}_target_increment"] = step[target_name] - previous
    return step


def wide_batch_features(step: pd.DataFrame) -> pd.DataFrame:
    fields = [
        field for field in (
            "time", "time_increment", "time_target", "time_target_increment",
            "energy", "energy_increment", "temp", "temp_target", "rotor", "rotor_target",
        ) if field in step.columns
    ]
    pieces = []
    for field in fields:
        pivot = step.pivot_table(
            index=["OrderID", "BatchNumber", "LabelGroupID"],
            columns=["GroupName", "StepNo"], values=field, aggfunc="last",
        )
        pivot.columns = [f"step_exec__{group}__S{int(number)}__{field}" for group, number in pivot.columns]
        pieces.append(pivot)
    wide = pd.concat(pieces, axis=1).reset_index()
    return wide


def within_order_associations(group: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    eligible = group.loc[group.groupby("OrderID")["LabelGroupID"].transform("size") >= 2].copy()
    y = eligible["MNY"] - eligible.groupby("OrderID")["MNY"].transform("mean")
    rows = []
    for feature in features:
        values = pd.to_numeric(eligible[feature], errors="coerce")
        x = values - values.groupby(eligible["OrderID"]).transform("mean")
        valid = x.notna() & y.notna()
        if int(valid.sum()) < 30 or x.loc[valid].nunique() < 3:
            continue
        rows.append(
            {
                "feature": feature,
                "group_rows": int(valid.sum()),
                "orders": int(eligible.loc[valid, "OrderID"].nunique()),
                "within_order_pearson": float(x.loc[valid].corr(y.loc[valid], method="pearson")),
                "within_order_spearman": float(x.loc[valid].corr(y.loc[valid], method="spearman")),
            }
        )
    result = pd.DataFrame(rows)
    if len(result):
        result["abs_spearman"] = result["within_order_spearman"].abs()
        result = result.sort_values(["abs_spearman", "feature"], ascending=[False, True])
    return result


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    phase = pd.read_csv(
        PHASE_SOURCE,
        usecols=["OrderID", "BatchNumber", "LabelGroupID", "CompoundName", "BatchStartTime"],
        low_memory=False,
    )
    phase = phase.loc[phase["CompoundName"].astype(str).eq(COMPOUND)].copy()
    phase["OrderID"], phase["BatchNumber"] = normalize(phase["OrderID"]), normalize(phase["BatchNumber"])
    keys = phase[["OrderID", "BatchNumber", "LabelGroupID", "BatchStartTime"]].drop_duplicates(["OrderID", "BatchNumber"])
    key_set = set(zip(keys["OrderID"], keys["BatchNumber"]))
    orders = set(keys["OrderID"])

    parts = []
    columns = ["OrderID", "BatchNumber", "GroupName", "StepNo", "VariablePath", "Target", "Value"]
    for path in sorted(BATCHDATA_DIR.glob("batchdata_orders_*.csv")):
        chunk = pd.read_csv(path, usecols=columns, dtype=str, low_memory=False)
        chunk["OrderID"] = normalize(chunk["OrderID"])
        selected = chunk.loc[chunk["OrderID"].isin(orders) & chunk["VariablePath"].isin(VARIABLES)].copy()
        if len(selected):
            selected["BatchNumber"] = normalize(selected["BatchNumber"])
            selected = selected.loc[[pair in key_set for pair in zip(selected["OrderID"], selected["BatchNumber"])]]
            if len(selected):
                parts.append(selected)
    raw = pd.concat(parts, ignore_index=True)
    raw = raw.merge(keys[["OrderID", "BatchNumber", "LabelGroupID"]], on=["OrderID", "BatchNumber"], how="inner", validate="many_to_one")
    step = build_batch_step_features(raw)
    step.to_csv(OUTPUT / "t15760_batch_step_execution_long.csv", index=False)
    batch = wide_batch_features(step)

    mst10 = "step_exec__AVR_MST__S10__time"
    mst11 = "step_exec__AVR_MST__S11__time"
    msb3 = "step_exec__AVR_MSB__S3__time"
    batch["tail_step11_actual_duration"] = pd.to_numeric(batch.get(mst11), errors="coerce") - pd.to_numeric(batch.get(mst10), errors="coerce")
    batch["msb_actual_duration"] = pd.to_numeric(batch.get(msb3), errors="coerce")
    valid_bottom = batch["msb_actual_duration"].fillna(0.0) > 1.0
    normal_tail = batch.loc[valid_bottom, "tail_step11_actual_duration"].dropna()
    tail_threshold = max(30.0, float(normal_tail.quantile(0.99)))
    batch["missing_bottom_execution"] = (~valid_bottom).astype(int)
    batch["long_tail_step11"] = batch["tail_step11_actual_duration"].gt(tail_threshold).astype(int)
    batch["post_pid_transfer_abnormal"] = ((batch["missing_bottom_execution"] > 0) | (batch["long_tail_step11"] > 0)).astype(int)
    batch.to_csv(OUTPUT / "t15760_batch_step_execution_wide.csv", index=False)

    group_source = pd.read_csv(GROUP_SOURCE, usecols=["LabelGroupID", "OrderID", "PalletID", "CompoundName", "MNY", "ProcessVersion", "lab_time"], low_memory=False)
    group = group_source.loc[group_source["CompoundName"].astype(str).eq(COMPOUND)].copy()
    group["OrderID"] = normalize(group["OrderID"])
    group["MNY"] = pd.to_numeric(group["MNY"], errors="coerce")
    numeric = [column for column in batch.columns if column.startswith("step_exec__") or column in {"tail_step11_actual_duration", "msb_actual_duration", "missing_bottom_execution", "long_tail_step11", "post_pid_transfer_abnormal"}]
    group_features = batch.groupby("LabelGroupID", as_index=False)[numeric].mean(numeric_only=True)
    count_features = batch.groupby("LabelGroupID", as_index=False).agg(
        physical_batch_count=("BatchNumber", "nunique"),
        abnormal_batch_count=("post_pid_transfer_abnormal", "sum"),
        abnormal_batch_rate=("post_pid_transfer_abnormal", "mean"),
        max_tail_step11_duration=("tail_step11_actual_duration", "max"),
        missing_bottom_batch_count=("missing_bottom_execution", "sum"),
    )
    group = group.merge(group_features, on="LabelGroupID", how="left", validate="one_to_one").merge(count_features, on="LabelGroupID", how="left", validate="one_to_one")
    group.to_csv(OUTPUT / "t15760_group_step_execution_features.csv", index=False)

    feature_columns = [column for column in group.columns if column.startswith("step_exec__") or column in {"tail_step11_actual_duration", "msb_actual_duration", "abnormal_batch_rate", "max_tail_step11_duration", "missing_bottom_batch_count"}]
    associations = within_order_associations(group, feature_columns)
    associations.to_csv(OUTPUT / "within_order_step_execution_associations.csv", index=False)

    order_stats = group.groupby("OrderID", as_index=False).agg(group_count=("LabelGroupID", "size"), mny_min=("MNY", "min"), mny_max=("MNY", "max"))
    order_stats["mny_range"] = order_stats["mny_max"] - order_stats["mny_min"]
    high_orders = order_stats.loc[order_stats["group_count"].ge(3) & order_stats["mny_range"].ge(8.0), "OrderID"]
    contrasts = []
    for order_id, order in group.loc[group["OrderID"].isin(high_orders)].groupby("OrderID"):
        low, high = order.loc[order["MNY"].idxmin()], order.loc[order["MNY"].idxmax()]
        row = {
            "OrderID": order_id, "low_group": low["LabelGroupID"], "high_group": high["LabelGroupID"],
            "low_mny": low["MNY"], "high_mny": high["MNY"], "delta_mny": high["MNY"] - low["MNY"],
            "low_physical_batches": low["physical_batch_count"], "high_physical_batches": high["physical_batch_count"],
            "low_abnormal_rate": low["abnormal_batch_rate"], "high_abnormal_rate": high["abnormal_batch_rate"],
            "low_max_tail_s": low["max_tail_step11_duration"], "high_max_tail_s": high["max_tail_step11_duration"],
            "low_missing_bottom_batches": low["missing_bottom_batch_count"], "high_missing_bottom_batches": high["missing_bottom_batch_count"],
        }
        for feature in feature_columns:
            row[f"delta__{feature}"] = pd.to_numeric(pd.Series([high.get(feature)]), errors="coerce").iloc[0] - pd.to_numeric(pd.Series([low.get(feature)]), errors="coerce").iloc[0]
        contrasts.append(row)
    contrasts = pd.DataFrame(contrasts).sort_values("delta_mny", ascending=False)
    contrasts.to_csv(OUTPUT / "high_variation_order_step_contrasts.csv", index=False)

    abnormal_groups = group["abnormal_batch_count"].fillna(0).gt(0)
    summary = {
        "contract": "T15760-STEP-EXECUTION-NATURAL-CONTRAST-V1",
        "batches": int(len(batch)), "groups": int(len(group)),
        "tail_step11_long_threshold_s": tail_threshold,
        "missing_bottom_batches": int(batch["missing_bottom_execution"].sum()),
        "long_tail_batches": int(batch["long_tail_step11"].sum()),
        "post_pid_transfer_abnormal_batches": int(batch["post_pid_transfer_abnormal"].sum()),
        "groups_with_abnormal_batch": int(abnormal_groups.sum()),
        "abnormal_group_mny_mean": float(group.loc[abnormal_groups, "MNY"].mean()),
        "normal_group_mny_mean": float(group.loc[~abnormal_groups, "MNY"].mean()),
        "top_within_order_step_associations": associations.head(25).to_dict("records"),
        "interpretation": "Step11 is trailing no-temperature-target execution; abnormality is a risk feature, not an adjustable PID action.",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2, default=str))


if __name__ == "__main__":
    main()
