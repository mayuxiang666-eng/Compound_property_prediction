from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyodbc


def configure_paths() -> Path:
    app_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(app_dir))
    pipeline_root = Path(os.environ.get("MOONEY_PIPELINE_ROOT", app_dir.parents[2]))
    sys.path.insert(1, str(pipeline_root))
    m_dir = pipeline_root / "Mooney_Prediction_Pipeline"
    if m_dir.exists():
        sys.path.insert(2, str(m_dir))
        for sub in ["data_processing", "model_analysis", "model_training", "feature_engineering"]:
            if (m_dir / sub).exists():
                sys.path.insert(3, str(m_dir / sub))
    return pipeline_root


PIPELINE_ROOT = configure_paths()

try:
    from Mooney_Prediction_Pipeline.feature_engineering.clustering import (
        cluster_silica_carbon_black,
    )
except ImportError:
    from feature_engineering.clustering import (
        cluster_silica_carbon_black,
    )

try:
    import predict_latest_sfe_order as source
except Exception:
    from Mooney_Prediction_Pipeline.model_analysis import predict_latest_sfe_order as source


LOG = logging.getLogger("mooney-v37-shadow")
REQUIRED_COLUMNS = [
    "OrderID", "BatchNumber", "CompoundName", "CompoundDescription",
    "OrderStartTime", "BatchWeight", "Top_Fill_Factor", "Bot_Fill_Factor",
    "Target_Temperature", "weight_pct_solid_elastomer",
    "weight_pct_natural_rubber", "weight_pct_silica", "weight_pct_oil",
    "weight_pct_silian", "weight_pct_carbon_black",
]


def read_mms_config() -> dict:
    path = Path(os.environ["MOONEY_MMS_CREDENTIALS_PATH"])
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)["HF_MMS"]


def connect_mms(database: str = "SFEPLANT") -> pyodbc.Connection:
    c = read_mms_config()
    connection = (
        f"DRIVER={c['driver']};SERVER={c['server']};DATABASE={database};"
        f"UID={c['username']};PWD={c['password']};"
        "Encrypt=yes;TrustServerCertificate=yes;"
    )
    return pyodbc.connect(connection, timeout=30)


def connect_result() -> pyodbc.Connection:
    connection = os.environ.get("MOONEY_RESULT_CONNECTION_STRING", "").strip()
    if not connection:
        raise RuntimeError("MOONEY_RESULT_CONNECTION_STRING is not configured")
    return pyodbc.connect(connection, timeout=15)


def patch_source_connection() -> None:
    # Existing query/extraction code uses this module-global connector. The
    # credential file itself remains outside the release package.
    source.connect_mms = connect_mms


def load_model() -> tuple[object, dict]:
    package_dir = Path(os.environ["MOONEY_MODEL_PACKAGE_DIR"])
    with (package_dir / "feature_metadata.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = joblib.load(package_dir / "hybrid_model.joblib")
    return model, metadata


def result_exists(order_id: str, batch_id: str) -> bool:
    with connect_result() as conn:
        row = conn.execute(
            "SELECT 1 FROM dbo.MooneyResult WHERE OrderID=? AND BatchID=?",
            order_id, batch_id,
        ).fetchone()
    return row is not None


def write_miss(row: pd.Series, reason: str, detail: str, model_version: str | None) -> None:
    sql = """
    INSERT INTO dbo.MooneyPredictMiss
    (OrderID, BatchID, CompName, ProductTime, MissReason, MissDetail,
     Retryable, RetryCount, ModelVersion)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    order_id = str(row.get("OrderID", ""))
    batch_id = str(int(row["BatchNumber"]))
    with connect_result() as conn:
        conn.execute(
            sql, order_id, batch_id, str(row.get("CompoundName", "")),
            row.get("OrderStartTime"), reason, detail, 1 if reason == "CURVE_INCOMPLETE" else 0,
            0, model_version,
        )
        conn.commit()


def write_prediction(row: pd.Series, features: dict, output: tuple, model_meta: dict) -> str:
    uncal, stage1, bias, stage2 = output
    order_id = str(row["OrderID"])
    batch_id = str(int(row["BatchNumber"]))
    raw = float(np.asarray(uncal).reshape(-1)[0])
    final = raw
    material = str(features.get("material_system", ""))
    phase = "oil_wet" if float(features.get("is_oil_loading_present", 0.0)) > 0 else "no_oil_dry"
    sql = """
    INSERT INTO dbo.MooneyResult
    (OrderID, BatchID, SourceBatchNumber, CompName, CompDescription, Mixer,
     MaterialSystem, PhaseRoute, ProductTime, ModelID, ModelVersion,
     Stage1Pred, Stage1bBias, Stage2Delta, RawPrediction,
     CalibrationOffset, FinalPrediction, LabStatus, FeatureMissingCount,
     CurvePointCount, FeatureBuilderVersion)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    values = (
        order_id, batch_id, str(row["BatchNumber"]), str(row.get("CompoundName", "")),
        str(row.get("CompoundDescription", "")), None, material, phase,
        row.get("OrderStartTime"), int(model_meta["ModelID"]), str(model_meta["ModelVersion"]),
        float(np.asarray(stage1).reshape(-1)[0]), float(np.asarray(bias).reshape(-1)[0]),
        float(np.asarray(stage2).reshape(-1)[0]), raw, 0.0, final, "PENDING",
        int(features.get("_missing_feature_count", 0)), int(features.get("_curve_point_count", 0)),
        "V3.7-shadow-feature-builder-v1",
    )
    try:
        with connect_result() as conn:
            conn.execute(sql, values)
            conn.commit()
        return "INSERTED"
    except pyodbc.Error as exc:
        if "duplicate" in str(exc).lower() or "unique" in str(exc).lower():
            return "ALREADY_EXISTS"
        raise


def one_pass(write: bool) -> dict:
    patch_source_connection()
    model, metadata = load_model()
    rows = source.query_latest_order_rows(candidate_count=20)
    if rows.empty:
        return {"candidates": 0, "inserted": 0, "missed": 0, "skipped": 0}

    inserted = missed = skipped = 0
    for _, row in rows.iterrows():
        order_id = str(row["OrderID"])
        batch_id = str(int(row["BatchNumber"]))
        if write and result_exists(order_id, batch_id):
            skipped += 1
            continue
        try:
            features = source.extract_features_for_row(row)
            features = cluster_silica_carbon_black(pd.DataFrame([features])).iloc[0].to_dict()
            feature_names = list(metadata["s1_features"]) + list(metadata["s2_features"])
            missing = [c for c in feature_names if c not in features or pd.isna(features[c])]
            model_input = pd.DataFrame([{c: features.get(c, 0.0) for c in feature_names}])
            model_input[feature_names] = model_input[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            model_input["CompoundName"] = features.get("CompoundName", row.get("CompoundName", ""))
            model_input["material_system"] = features.get("material_system", "CarbonBlack")
            output = model.predict(model_input, cluster_col="material_system")
            features["_missing_feature_count"] = len(missing)
            features["_curve_point_count"] = int(features.get("idx_total_length", 0))
            if write:
                meta = {"ModelID": 2, "ModelVersion": metadata.get("model_version", "V3.7.0")}
                status = write_prediction(row, features, output, meta)
                inserted += int(status == "INSERTED")
            else:
                LOG.info("DRY_RUN order=%s batch=%s missing_features=%s", order_id, batch_id, len(missing))
        except Exception as exc:  # one bad batch must not stop the polling loop
            missed += 1
            LOG.exception("batch_failed order=%s batch=%s", order_id, batch_id)
            if write:
                write_miss(row, "FEATURE_OR_PREDICT_FAILED", str(exc)[:1800], metadata.get("model_version"))
    return {"candidates": len(rows), "inserted": inserted, "missed": missed, "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    if args.write and os.environ.get("MOONEY_RELEASE_TYPE") != "SHADOW":
        raise RuntimeError("Write mode requires MOONEY_RELEASE_TYPE=SHADOW")
    while True:
        started = time.perf_counter()
        LOG.info("poll=%s", one_pass(write=args.write))
        if args.once:
            return
        time.sleep(max(0.0, float(os.environ.get("PREDICT_POLL_SECONDS", "2")) - (time.perf_counter() - started)))


if __name__ == "__main__":
    main()

