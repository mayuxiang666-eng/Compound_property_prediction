"""
Compound Tracer Module: Trace Final Mix (终胶车次) to Masterbatch (母胶车次)
and extract synchronized mixing process curves from SFE database.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import db_client


def decode_curve_hex(hex_str: Optional[str]) -> List[int]:
    """Decode a 4-hex-char 16-bit integer string into numerical time series."""
    if not hex_str or not isinstance(hex_str, str):
        return []
    return [int(hex_str[i:i+4], 16) for i in range(0, len(hex_str), 4) if len(hex_str[i:i+4]) == 4]


def trace_final_mix_to_masterbatch(
    order_id: str, 
    batch_number: int, 
    database: str = "HESFESFEPLANT"
) -> Dict[str, Any]:
    """
    Given a Final Mix OrderID and BatchNumber, trace back to the exact
    Masterbatch (母胶) Order, Pallet(s), and Batch number ranges.
    """
    # 1. Query Final Mix Batch Header
    sql_fm = f"""
    SELECT 
        b.OrderID, b.BatchNumber, b.EquipmentID, b.BatchStartTime, b.BatchEndTime,
        b.BatchWeight, b.PalletID, o.CompoundCode, o.CompoundName, o.ProdVersionText
    FROM BatchHeader b
    LEFT JOIN Orders o ON b.OrderID = o.OrderID
    WHERE b.OrderID = '{order_id}' AND b.BatchNumber = {batch_number}
    """
    df_fm = db_client.query_df(sql_fm, database=database)
    if df_fm.empty:
        raise ValueError(f"Final Mix batch not found for OrderID '{order_id}', Batch {batch_number}")
    fm_info = df_fm.iloc[0].to_dict()

    # 2. Query Materials charged into this Final Mix Batch (Focus on Masterbatch M1/MB)
    sql_mat = f"""
    SELECT 
        OrderID, BatchNumber, SequenceNumber, MaterialCode, MaterialName,
        ChargeID, TargetWeight, ActualWeight, ConsumedWeightFromLot
    FROM BatchMaterials
    WHERE OrderID = '{order_id}' AND BatchNumber = {batch_number}
      AND (MaterialName LIKE 'M1-%' OR MaterialName LIKE 'MB-%' OR MaterialCode LIKE '9922A%')
    ORDER BY SequenceNumber, ChargeID
    """
    df_mat = db_client.query_df(sql_mat, database=database)

    # 3. For each ChargeID (which corresponds to PalletID), trace back to Masterbatch BatchHeader
    mb_traces = []
    charge_ids = df_mat['ChargeID'].dropna().unique().tolist()
    
    if charge_ids:
        in_clause = ",".join([f"'{c}'" for c in charge_ids])
        sql_mb = f"""
        SELECT 
            b.OrderID AS MB_OrderID,
            b.PalletID AS MB_PalletID,
            MIN(b.BatchNumber) AS MB_StartBatch,
            MAX(b.BatchNumber) AS MB_EndBatch,
            COUNT(*) AS MB_BatchCount,
            MIN(b.BatchStartTime) AS MB_FirstBatchStart,
            MAX(b.BatchEndTime) AS MB_LastBatchEnd,
            b.EquipmentID AS MB_Equipment,
            o.CompoundCode AS MB_CompoundCode,
            o.CompoundName AS MB_CompoundName
        FROM BatchHeader b
        LEFT JOIN Orders o ON b.OrderID = o.OrderID
        WHERE b.PalletID IN ({in_clause})
        GROUP BY b.OrderID, b.PalletID, b.EquipmentID, o.CompoundCode, o.CompoundName
        ORDER BY b.OrderID, MB_StartBatch
        """
        df_mb = db_client.query_df(sql_mb, database=database)
        if not df_mb.empty and fm_info.get("BatchStartTime"):
            fm_start = pd.to_datetime(fm_info["BatchStartTime"])
            df_mb["AgingHours"] = (fm_start - pd.to_datetime(df_mb["MB_LastBatchEnd"])).dt.total_seconds() / 3600.0
            df_mb["AgingHours"] = df_mb["AgingHours"].round(2)
        mb_traces = df_mb.to_dict(orient="records")

    return {
        "final_mix": fm_info,
        "charged_materials": df_mat.to_dict(orient="records"),
        "masterbatch_links": mb_traces
    }


def get_batch_curves(
    order_id: str, 
    batch_number: int, 
    database: str = "HESFESFEPLANT"
) -> pd.DataFrame:
    """
    Retrieve and decode the mixing process curve (Temperature, Power, Speed, Ram)
    for any batch (Final Mix or Masterbatch).
    """
    sql = f"""
    SELECT OrderID, BatchNumber, EquipmentID, Curve1, Curve2, Curve3, Curve5, Curve6
    FROM BatchCurve
    WHERE OrderID = '{order_id}' AND BatchNumber = {batch_number}
    """
    df = db_client.query_df(sql, database=database)
    if df.empty:
        return pd.DataFrame()

    row = df.iloc[0]
    temp = decode_curve_hex(row.get("Curve1"))
    power = decode_curve_hex(row.get("Curve2"))
    ram = decode_curve_hex(row.get("Curve3"))
    torque = decode_curve_hex(row.get("Curve5"))
    speed = decode_curve_hex(row.get("Curve6"))

    max_len = max(len(temp), len(power), len(ram), len(torque), len(speed), 0)
    
    # Pad series to equal length
    def pad(lst, target_len):
        return lst + [None] * (target_len - len(lst))

    return pd.DataFrame({
        "time_step_sec": list(range(1, max_len + 1)),
        "temperature_c": pad(temp, max_len),
        "power_kw": pad(power, max_len),
        "ram_position": pad(ram, max_len),
        "torque_energy": pad(torque, max_len),
        "rotor_speed_rpm": pad(speed, max_len),
    })


def trace_final_mix_with_anomaly(
    order_id: str, 
    batch_number: int, 
    database: str = "HESFESFEPLANT"
) -> Dict[str, Any]:
    """
    Combined Pipeline:
    1. Traces Final Mix (终胶) to all corresponding Masterbatches (母胶) via SFE Cloud DB.
    2. Decorates both Final Mix and Masterbatch records with AI/Expert anomaly judgments
       from the Mixing Curve Analysis System (JudgeState, RamPosition, Power alerts).
    """
    import mixing_system_client

    trace_res = trace_final_mix_to_masterbatch(order_id, batch_number, database=database)

    # 1. Fetch anomaly judgment for Final Mix batch
    fm_anomaly = mixing_system_client.get_batch_anomaly(order_id, batch_number)
    trace_res["final_mix_anomaly"] = {
        "is_normal": fm_anomaly.get("is_normal"),
        "judge_state": fm_anomaly.get("judge_state"),
        "judge_reason": fm_anomaly.get("judge_reason"),
        "ram_judge_state": fm_anomaly.get("ram_judge_state"),
    }

    # 2. Fetch anomaly judgment for each matched Masterbatch range
    decorated_mb = []
    for mb in trace_res.get("masterbatch_links", []):
        mb_copy = dict(mb)
        mb_order = str(mb.get("MB_OrderID"))
        start_batch = mb.get("MB_StartBatch")
        mb_anom = mixing_system_client.get_batch_anomaly(mb_order, start_batch)
        mb_copy["MB_StartBatch_Anomaly"] = {
            "is_normal": mb_anom.get("is_normal"),
            "judge_state": mb_anom.get("judge_state"),
            "judge_reason": mb_anom.get("judge_reason"),
            "ram_judge_state": mb_anom.get("ram_judge_state"),
        }
        decorated_mb.append(mb_copy)

    trace_res["masterbatch_links"] = decorated_mb
    return trace_res


def get_pallet_mooney_and_lab_quality(
    order_id: str, 
    pallet_id: str, 
    database: str = "HESFESFEPLANT"
) -> Dict[str, Any]:
    """
    Retrieve laboratory testing results (Mooney ML1+3, Rheometer, etc.)
    from Redshift he_datamarts.compound_excellence_datamart, matched with
    SFE PalletSample sampling metadata.
    """
    # 1. Fetch samples taken from this pallet in SFE
    sql_samples = f"SELECT SampleID, PalletID, Timestamp FROM PalletSample WHERE PalletID = '{pallet_id}'"
    df_samples = db_client.query_df(sql_samples, database=database)
    sample_ids = df_samples['SampleID'].tolist() if not df_samples.empty else []

    # 2. Query Redshift compound_excellence_datamart
    sql_lab = f"""
    SELECT 
        order_id, pallet_id, sample_id, test_variable, test_name, 
        test_result, uom, test_status, test_result_start_time
    FROM he_datamarts.compound_excellence_datamart
    WHERE order_id = '{order_id}' AND pallet_id = '{pallet_id}'
    ORDER BY test_result_start_time
    """
    try:
        df_lab = db_client.query_redshift_df(sql_lab)
    except Exception as exc:
        return {
            "error": f"Failed to query Redshift: {exc}",
            "mooney_ml1_3": None,
            "samples_in_sfe": sample_ids,
            "lab_tests": []
        }

    # Extract Mooney ML1+3
    mooney_row = df_lab[df_lab['test_variable'] == 'ML1+3'] if not df_lab.empty else pd.DataFrame()
    mooney_val = float(mooney_row.iloc[0]['test_result']) if (not mooney_row.empty and pd.notnull(mooney_row.iloc[0]['test_result'])) else None
    mooney_status = mooney_row.iloc[0]['test_status'] if not mooney_row.empty else None
    mooney_time = str(mooney_row.iloc[0]['test_result_start_time']) if not mooney_row.empty else None
    mooney_sample = str(mooney_row.iloc[0]['sample_id']) if not mooney_row.empty else None

    return {
        "pallet_id": pallet_id,
        "order_id": order_id,
        "samples_in_sfe": sample_ids,
        "mooney_ml1_3": mooney_val,
        "mooney_unit": "MU" if mooney_val is not None else None,
        "mooney_status": mooney_status,
        "mooney_sample_id": mooney_sample,
        "mooney_test_time": mooney_time,
        "total_lab_tests_count": len(df_lab),
        "lab_tests": df_lab.to_dict(orient="records") if not df_lab.empty else []
    }


def trace_final_mix_full_pipeline(
    order_id: str, 
    batch_number: int, 
    database: str = "HESFESFEPLANT"
) -> Dict[str, Any]:
    """
    Complete Full-Pipeline Trace:
    1. SFE Cloud DB: Final Mix batch metadata & PalletID.
    2. Redshift DataMart: Final Mix Pallet Mooney & Lab Quality results (via PalletSample + compound_excellence_datamart).
    3. SFE Cloud DB: Upstream Masterbatch charged pallets and batches (ChargeID == PalletID).
    4. Mixing Curve API: Final Mix & Masterbatch AI anomaly evaluations.
    """
    trace_res = trace_final_mix_with_anomaly(order_id, batch_number, database=database)
    pallet_id = trace_res.get("final_mix", {}).get("PalletID")
    
    if pallet_id:
        lab_qual = get_pallet_mooney_and_lab_quality(order_id, pallet_id, database=database)
        trace_res["final_mix_lab_quality"] = lab_qual
    else:
        trace_res["final_mix_lab_quality"] = None

    return trace_res


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Testing Full Pipeline Trace: Final Mix (8184718-1) -> Mooney -> Masterbatches ===")
    res = trace_final_mix_full_pipeline("8184718", 1)
    
    print("\n[1] Final Mix Info & Anomaly Status:")
    for k, v in res["final_mix"].items():
        print(f"  {k}: {v}")
    print("  --> 过程异常判定:", res["final_mix_anomaly"])

    print("\n[2] Final Mix Lab Quality (Mooney Viscosity from Redshift):")
    lq = res.get("final_mix_lab_quality", {})
    print(f"  托盘号: {lq.get('pallet_id')}")
    print(f"  SFE取样批号: {lq.get('samples_in_sfe')}")
    print(f"  终胶门尼值 (ML1+3): {lq.get('mooney_ml1_3')} {lq.get('mooney_unit')} (判定状态: {lq.get('mooney_status')})")
    print(f"  Redshift化验取样ID: {lq.get('mooney_sample_id')}")
    print(f"  化验出样时间: {lq.get('mooney_test_time')}")

    print("\n[3] Charged Mother Compound Feeds:")
    for row in res["charged_materials"]:
        print(f"  Material: {row['MaterialName']} ({row['MaterialCode']}), ChargeID: {row['ChargeID']}, Consumed: {row['ConsumedWeightFromLot']} kg")

    print("\n[4] Matched Mother Compound Batches & Their Anomaly Status:")
    for mb in res["masterbatch_links"]:
        print(f"  MB Order: {mb['MB_OrderID']}, Pallet: {mb['MB_PalletID']}, Batches: {mb['MB_StartBatch']} ~ {mb['MB_EndBatch']}")
        print(f"      异常判定: {mb.get('MB_StartBatch_Anomaly')}")

    print("\n[5] Decoded Curve Preview:")
    curve_df = get_batch_curves("8184718", 1)
    print(curve_df.head(6))

