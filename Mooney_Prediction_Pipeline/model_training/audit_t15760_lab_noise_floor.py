"""Audit T15760 Lab revision/retest and multi-sample disagreement."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
SOURCE = BASE / "lab_ms13_9200_raw.csv"
OUTPUT = BASE / "t15760_lab_noise_floor"


def describe(values: pd.Series) -> dict:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return {"n": 0}
    return {
        "n": int(len(values)), "mean": float(values.mean()), "median": float(values.median()),
        "p75": float(values.quantile(0.75)), "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)), "max": float(values.max()),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(SOURCE, low_memory=False)
    raw = raw.loc[
        raw["compound_name"].astype(str).str.contains("T15760", regex=False)
        & raw["pallet_id"].notna()
        & pd.to_numeric(raw["disposition_code"], errors="coerce").isin([1, 11])
        & raw["status_valid"].astype(str).str.lower().eq("true")
    ].copy()
    raw["test_result"] = pd.to_numeric(raw["test_result"], errors="coerce")
    raw["last_modified_timestamp"] = pd.to_datetime(raw["last_modified_timestamp"], errors="coerce")
    raw["test_result_end_time"] = pd.to_datetime(raw["test_result_end_time"], errors="coerce")
    raw = raw.dropna(subset=["test_result"])
    raw["LabelGroupID"] = raw["order_id"].astype(str).str.replace(r"\.0$", "", regex=True) + "|" + raw["pallet_id"].astype(str).str.replace(r"\.0$", "", regex=True)
    sample_key = ["order_id", "pallet_id", "sample_id"]

    revision = raw.groupby(sample_key, dropna=False).agg(
        rows=("test_result", "size"), unique_results=("test_result", "nunique"),
        min_result=("test_result", "min"), max_result=("test_result", "max"),
        retest_levels=("retest_number_numeric", "nunique"),
    ).reset_index()
    revision["result_span"] = revision["max_result"] - revision["min_result"]
    revision = revision.loc[revision["unique_results"].gt(1)].copy()
    revision.to_csv(OUTPUT / "t15760_changed_sample_revisions.csv", index=False)

    latest = raw.sort_values(["last_modified_timestamp", "test_result_end_time"], kind="stable").drop_duplicates(sample_key, keep="last")
    group = latest.groupby("LabelGroupID").agg(
        sample_count=("sample_id", "nunique"), unique_results=("test_result", "nunique"),
        min_result=("test_result", "min"), max_result=("test_result", "max"),
        mean_result=("test_result", "mean"), std_result=("test_result", lambda s: float(s.std(ddof=0))),
    ).reset_index()
    group["result_span"] = group["max_result"] - group["min_result"]
    multi = group.loc[group["sample_count"].gt(1)].copy()
    multi.to_csv(OUTPUT / "t15760_multi_sample_group_disagreement.csv", index=False)

    paired_abs = []
    for _, part in latest.groupby("LabelGroupID"):
        values = part["test_result"].dropna().to_numpy(dtype=float)
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                paired_abs.append(abs(values[i] - values[j]))
    summary = {
        "contract": "T15760-LAB-NOISE-FLOOR-AUDIT-V1",
        "normal_disposition_codes": ["001", "011"],
        "raw_rows": int(len(raw)), "sample_keys": int(raw.groupby(sample_key).ngroups),
        "changed_sample_revision_keys": int(len(revision)),
        "changed_sample_revision_span": describe(revision["result_span"]),
        "label_groups": int(len(group)), "multi_sample_groups": int(len(multi)),
        "multi_sample_group_span": describe(multi["result_span"]),
        "within_group_pairwise_absolute_difference": describe(pd.Series(paired_abs, dtype=float)),
        "interpretation_guard": "Multi-sample disagreement includes sampling heterogeneity plus laboratory repeatability; it is an empirical lower-bound diagnostic, not pure instrument Gage R&R.",
        "target_2_5_mu_status": "REQUIRES_GAGE_RR_OR_CONTROLLED_REPLICATE_CONFIRMATION",
        "database_access_during_run": False, "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
