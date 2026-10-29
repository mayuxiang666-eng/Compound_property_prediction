"""
Mooney Prediction & Process Curve Analysis Engine
Performs real-time database queries, curve decoding, physical feature extraction,
multi-batch Mooney delta prediction, and dual-batch physical comparison.
"""

from typing import Dict, Any, List, Optional, Tuple
import time
import numpy as np
import pandas as pd
import db_client

# Stage definitions (11 windows of ~22 seconds)
STEP_DEFINITIONS = [
    {"step": 1,  "window": "0-22s",    "name": "上密炼开盖投母胶吃料", "desc": "母胶/回炼胶投入，转子提速吃料，机身与物料温差决定初始软化速度与剪切阻力"},
    {"step": 2,  "window": "22-44s",   "name": "一段下顶栓压合初炼",   "desc": "顶栓加压压合，高剪切压实破碎，早期胶温上升与机械能输入"},
    {"step": 3,  "window": "44-66s",   "name": "提速40rpm主剪切混炼", "desc": "主剪切峰值段，冷胶撕扯峰值负荷极高，热胶阻力明显降低"},
    {"step": 4,  "window": "66-88s",   "name": "提栓投小料二次压合",   "desc": "顶栓提升加药（硫黄/促进剂），二次下栓压合，温升略受加药吸热影响"},
    {"step": 5,  "window": "88-110s",  "name": "二段高功率捏合分散",   "desc": "药剂在胶体中微观分散，高功率捏合，决定终胶结合胶形成与黏度平台"},
    {"step": 6,  "window": "110-132s", "name": "降速15rpm均化控温",   "desc": "转子降速防焦控温，低剪切均化混炼"},
    {"step": 7,  "window": "132-154s", "name": "上密排料转入下密炼",   "desc": "上密炼卸料门打开，胶料落入下密炼机，转接温度与排料残留"},
    {"step": 8,  "window": "154-176s", "name": "下密炼初期吃料运转",   "desc": "下密炼转子承接胶料，18~20rpm 低温均化"},
    {"step": 9,  "window": "176-198s", "name": "下密炼主均化段",       "desc": "下密炼恒速持续混炼，微观分散度最终成型"},
    {"step": 10, "window": "198-220s", "name": "下密炼末段控温排胶",   "desc": "接近 100~105°C 排胶阈值，准备卸出至开炼线"},
    {"step": 11, "window": "220-242s", "name": "下密收尾与排料过渡",   "desc": "下密炼翻斗/卸料完成，由负载转入空转阶段"}
]

# Model weights derived from historical clean order pairs
MODEL_WEIGHTS = {
    "active_huber": {
        "b0": -1.1456,
        "w_T_w01": 0.1887,
        "w_P_w05": 0.0616,
        "w_T_w05": -0.0053,
        "w_P_w06": 0.0111,
        "w_P_active": 0.0994
    },
    "phase4_huber": {
        "b0": -1.7461,
        "w_T_w01": 0.1291,
        "w_P_w05": 0.0623,
        "w_P_w06": 0.0245,
        "w_P_low": 0.0327
    },
    "raw_huber": {
        "b0": -1.1456,
        "w_T_w01": 0.1887,
        "w_P_w05": 0.0616,
        "w_T_w05": -0.0053,
        "w_P_w06": 0.0111,
        "w_Pmean": 0.0994
    },
    "model4_duration": {
        "b0": -1.3008,
        "w_Ti": -0.0034,
        "w_T_w01": 0.2846,
        "w_P_up_mean": -0.0821,
        "w_P_low_mean": -0.0287,
        "w_Pmean": 0.2557
    }
}

# Cache for loaded orders to make UI navigation instant
_ORDER_CACHE: Dict[str, Dict[str, Any]] = {}


def decode_clean_curve(hex_str: Optional[str]) -> Tuple[np.ndarray, int]:
    """Decode hex curve string, handle 0x8000 and signed integers."""
    if not hex_str or not isinstance(hex_str, str):
        return np.array([], dtype=np.float64), 0
    raw = [int(hex_str[i:i+4], 16) for i in range(0, len(hex_str), 4) if len(hex_str[i:i+4]) == 4]
    out = []
    n8000 = 0
    for v in raw:
        if v == 0x8000:
            n8000 += 1
            break
        elif v > 0x8000:
            out.append(max(0.0, float(v - 0x10000)))
        else:
            out.append(float(v))
    return np.array(out, dtype=np.float64), n8000


def safe_mean(arr, default: float = 0.0) -> float:
    """Compute mean safely, preventing NaN / Inf or empty slice warnings."""
    if arr is None or len(arr) == 0:
        return float(default)
    try:
        m = np.nanmean(arr)
        if np.isnan(m) or np.isinf(m):
            return float(default)
        return float(m)
    except Exception:
        return float(default)


def clean_float(val: Any, default: float = 0.0) -> float:
    """Ensure float is JSON compliant (not NaN or Inf)."""
    if val is None:
        return float(default)
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return float(default)
        return f
    except Exception:
        return float(default)


def analyze_order_full(order_id: str, baseline_batch: int = 1, force_refresh: bool = False) -> Dict[str, Any]:
    """
    Complete analysis pipeline for an order:
    1. Fetch Order Metadata (BatchHeader, Orders)
    2. Fetch Lab Mooney Tests (he_datamarts.compound_excellence_datamart)
    3. Fetch all BatchCurves (BatchCurve)
    4. Decode & slice features for all batches
    5. Calculate Mooney delta predictions for all batches relative to baseline_batch
    """
    order_id = str(order_id).strip()
    cache_key = f"{order_id}_{baseline_batch}"
    if not force_refresh and cache_key in _ORDER_CACHE:
        return _ORDER_CACHE[cache_key]

    t_start = time.time()

    # 1. Fetch Order Metadata
    sql_meta = f"""
    SELECT 
        b.OrderID,
        b.BatchNumber,
        b.EquipmentID,
        b.BatchStartTime,
        b.BatchEndTime,
        b.BatchCycleTimeSeconds AS CycleTime,
        b.MixingTimeSeconds AS MixingTime,
        b.BatchWeight,
        b.PalletID,
        o.CompoundCode,
        o.CompoundName,
        o.ProdVersionText
    FROM BatchHeader b
    LEFT JOIN Orders o ON b.OrderID = o.OrderID
    WHERE b.OrderID = '{order_id}'
    ORDER BY b.BatchNumber
    """
    df_meta = db_client.query_df(sql_meta, database="HESFESFEPLANT")
    if df_meta.empty:
        raise ValueError(f"工单 '{order_id}' 在 SFE 数据库 BatchHeader 中未找到任何车次记录。")

    total_batches = len(df_meta)
    pallet_list = df_meta['PalletID'].dropna().unique().tolist()
    total_pallets = len(pallet_list)
    min_pallet = min(pallet_list) if pallet_list else ""
    max_pallet = max(pallet_list) if pallet_list else ""
    equipment_id = df_meta['EquipmentID'].iloc[0] if 'EquipmentID' in df_meta.columns else "Unknown"
    compound_code = df_meta['CompoundCode'].iloc[0] if 'CompoundCode' in df_meta.columns and pd.notna(df_meta['CompoundCode'].iloc[0]) else "Unknown"
    compound_name = df_meta['CompoundName'].iloc[0] if 'CompoundName' in df_meta.columns and pd.notna(df_meta['CompoundName'].iloc[0]) else "Unknown"
    start_time_str = str(df_meta['BatchStartTime'].min())
    end_time_str = str(df_meta['BatchEndTime'].max())

    # Build batch-to-pallet mapping
    batch_pallet_map = {}
    pallet_batches_map = {}
    for _, r in df_meta.iterrows():
        b_num = int(r['BatchNumber'])
        p_id = str(r['PalletID']) if pd.notna(r['PalletID']) else ""
        batch_pallet_map[b_num] = p_id
        if p_id:
            pallet_batches_map.setdefault(p_id, []).append(b_num)

    # 2. Query Lab Mooney Tests from Redshift
    lab_tests = []
    has_lab_tests = False
    try:
        sql_lab = f"""
        SELECT order_id, pallet_id, sample_id, test_variable, test_result, test_status
        FROM he_datamarts.compound_excellence_datamart
        WHERE order_id = '{order_id}' AND test_variable = 'ML1+3'
        ORDER BY pallet_id, sample_id
        """
        df_lab = db_client.query_redshift_df(sql_lab)
        if not df_lab.empty:
            has_lab_tests = True
            for _, r in df_lab.iterrows():
                p_id = str(r['pallet_id'])
                sample_id = str(r['sample_id'])
                res = float(r['test_result'])
                status = str(r['test_status'])

                # Determine mapped batch
                mapped_batch = None
                mapping_type = "中间托盘样本"
                if p_id == min_pallet or sample_id == '001':
                    mapped_batch = 1
                    mapping_type = "第1车 (首车1:1)"
                elif p_id == max_pallet:
                    mapped_batch = total_batches
                    mapping_type = f"第{total_batches}车 (尾车1:1)"
                elif p_id in pallet_batches_map:
                    batches = pallet_batches_map[p_id]
                    mapped_batch = batches[len(batches) // 2]
                    mapping_type = f"托盘 {p_id} (车次 {min(batches)}~{max(batches)})"

                lab_tests.append({
                    "pallet_id": p_id,
                    "sample_id": sample_id,
                    "test_variable": "ML1+3",
                    "mooney_value": res,
                    "status": status,
                    "mapped_batch": mapped_batch,
                    "mapping_type": mapping_type
                })
    except Exception as e:
        # Gracefully handle lab query errors or missing tables
        print(f"Warning: Lab query failed: {e}")
        has_lab_tests = False

    # 3. Query all BatchCurves from SFE database
    sql_curves = f"""
    SELECT BatchNumber, Curve1, Curve2, Curve3, Curve5, Curve6, Curve7
    FROM BatchCurve
    WHERE OrderID = '{order_id}'
    ORDER BY BatchNumber
    """
    df_curves = db_client.query_df(sql_curves, database="HESFESFEPLANT")
    if df_curves.empty:
        raise ValueError(f"工单 '{order_id}' 在 SFE 数据库 BatchCurve 中未找到密炼曲线数据。")

    # 4. Decode curves and extract physical features for each batch
    batch_features: Dict[int, Dict[str, Any]] = {}
    batch_curves_cache: Dict[int, Dict[str, np.ndarray]] = {}

    for _, r in df_curves.iterrows():
        b = int(r['BatchNumber'])
        c1, _ = decode_clean_curve(r.get('Curve1'))  # Temp (°C)
        c2, _ = decode_clean_curve(r.get('Curve2'))  # Power (kW)
        c3, _ = decode_clean_curve(r.get('Curve3'))  # Energy (kWh * 10)
        c5, _ = decode_clean_curve(r.get('Curve5'))  # Torque
        c6, _ = decode_clean_curve(r.get('Curve6'))  # Speed (rpm)
        c7, _ = decode_clean_curve(r.get('Curve7'))  # Ram position (mm)

        L = min(len(c1), len(c2), len(c3), len(c6))
        if L < 50:
            continue

        temp_arr = c1[:L]
        power_arr = c2[:L]
        energy_arr = c3[:L] / 10.0
        speed_arr = c6[:L]
        ram_arr = c7[:min(len(c7), L)] if len(c7) > 0 else np.array([])

        # Upper mixer discharge time detection
        if len(c7) > 0:
            if 100 <= len(c7) <= 165:
                T_upper = len(c7)
            elif len(c7) > 165:
                lift_idx = None
                for idx in range(120, min(165, len(c7))):
                    if c7[idx] >= 1000:
                        lift_idx = idx
                        break
                T_upper = lift_idx if lift_idx is not None else 145
            else:
                T_upper = len(c7)
        else:
            T_upper = 145
        T_upper = min(T_upper, max(50, L - 10))

        # Detect Lower Mixer discharge / empty idle start
        # Discharge is where power drops to empty idle (< 35 kW) and speed >= 25 rpm
        p_active_end = L
        for t in range(min(145, L - 5), L - 3):
            if power_arr[t] < 35.0 and speed_arr[t] >= 25.0:
                if np.all(power_arr[t:min(t+3, L)] < 38.0):
                    p_active_end = t
                    break

        # Cache curve arrays for UI dual-batch comparison
        batch_curves_cache[b] = {
            "temp": temp_arr,
            "power": power_arr,
            "energy": energy_arr,
            "speed": speed_arr,
            "ram": ram_arr,
            "L": L,
            "T_upper": T_upper,
            "T_active": p_active_end
        }

        # Calculate active powers safely
        p_raw_mean = safe_mean(power_arr, default=250.0)
        p_active_mean = safe_mean(power_arr[:p_active_end], default=p_raw_mean)
        p_up_mean = safe_mean(power_arr[:T_upper], default=p_active_mean)
        
        if p_active_end > T_upper + 5:
            low_slice = power_arr[T_upper:p_active_end]
        else:
            low_slice = power_arr[max(0, p_active_end - 60):p_active_end]
        p_low_mean = safe_mean(low_slice, default=p_active_mean)

        t0 = safe_mean(temp_arr[:5], default=temp_arr[0] if len(temp_arr) > 0 else 70.0)
        t_discharge = float(temp_arr[p_active_end-1]) if (0 < p_active_end <= len(temp_arr)) else float(temp_arr[-1])
        t_max = float(np.max(temp_arr)) if len(temp_arr) > 0 else t0
        e_discharge = float(energy_arr[p_active_end-1]) if (0 < p_active_end <= len(energy_arr)) else float(energy_arr[-1])

        # 11-step window features
        w_feats = {}
        for w in range(11):
            s_idx = w * 22
            e_idx = min((w + 1) * 22, L)
            if s_idx < L and e_idx > s_idx:
                w_feats[f"T_w{w:02d}"] = safe_mean(temp_arr[s_idx:e_idx], default=temp_arr[-1])
                w_feats[f"P_w{w:02d}"] = safe_mean(power_arr[s_idx:e_idx], default=power_arr[-1])
                w_feats[f"S_w{w:02d}"] = safe_mean(speed_arr[s_idx:e_idx], default=speed_arr[-1])
                if len(ram_arr) > s_idx:
                    w_feats[f"R_w{w:02d}"] = safe_mean(ram_arr[s_idx:min(e_idx, len(ram_arr))], default=0.0)
                else:
                    w_feats[f"R_w{w:02d}"] = 0.0
            else:
                w_feats[f"T_w{w:02d}"] = float(temp_arr[-1])
                w_feats[f"P_w{w:02d}"] = float(power_arr[-1])
                w_feats[f"S_w{w:02d}"] = float(speed_arr[-1])
                w_feats[f"R_w{w:02d}"] = 0.0

        batch_features[b] = {
            "batch_number": b,
            "pallet_id": batch_pallet_map.get(b, ""),
            "T0": t0,
            "T_discharge": t_discharge,
            "T_max": t_max,
            "T_upper": T_upper,
            "T_active": p_active_end,
            "T_total": L,
            "P_active": p_active_mean,
            "P_up": p_up_mean,
            "P_low": p_low_mean,
            "P_raw": p_raw_mean,
            "E_discharge": e_discharge,
            **w_feats
        }

    # Verify baseline batch
    if baseline_batch not in batch_features:
        baseline_batch = min(batch_features.keys()) if batch_features else 1
    base_f = batch_features[baseline_batch]

    # Baseline lab Mooney if available
    base_lab_mooney = None
    for lt in lab_tests:
        if lt["mapped_batch"] == baseline_batch:
            base_lab_mooney = lt["mooney_value"]
            break

    # 5. Predict Mooney Delta for all batches relative to baseline
    full_batch_results = []
    sorted_batches = sorted(batch_features.keys())

    for b in sorted_batches:
        f = batch_features[b]
        dT_w01 = f["T_w01"] - base_f["T_w01"]
        dP_w05 = f["P_w05"] - base_f["P_w05"]
        dT_w05 = f["T_w05"] - base_f["T_w05"]
        dP_w06 = f["P_w06"] - base_f["P_w06"]
        dP_active = f["P_active"] - base_f["P_active"]
        dP_low = f["P_low"] - base_f["P_low"]
        dP_raw = f["P_raw"] - base_f["P_raw"]

        # If baseline batch itself, delta is exactly 0
        if b == baseline_batch:
            pred_active = 0.0
            pred_phase4 = 0.0
            pred_raw = 0.0
            pred_m4 = 0.0
            pred_ensemble = 0.0
        else:
            # Model 1: Active Huber
            w_act = MODEL_WEIGHTS["active_huber"]
            pred_active = (
                w_act["b0"] +
                w_act["w_T_w01"] * dT_w01 +
                w_act["w_P_w05"] * dP_w05 +
                w_act["w_T_w05"] * dT_w05 +
                w_act["w_P_w06"] * dP_w06 +
                w_act["w_P_active"] * dP_active
            )

            # Model 2: 4-Phase Huber
            w_p4 = MODEL_WEIGHTS["phase4_huber"]
            pred_phase4 = (
                w_p4["b0"] +
                w_p4["w_T_w01"] * dT_w01 +
                w_p4["w_P_w05"] * dP_w05 +
                w_p4["w_P_w06"] * dP_w06 +
                w_p4["w_P_low"] * dP_low
            )

            # Model 3: Raw Fixed Window Huber
            w_raw = MODEL_WEIGHTS["raw_huber"]
            pred_raw = (
                w_raw["b0"] +
                w_raw["w_T_w01"] * dT_w01 +
                w_raw["w_P_w05"] * dP_w05 +
                w_raw["w_T_w05"] * dT_w05 +
                w_raw["w_P_w06"] * dP_w06 +
                w_raw["w_Pmean"] * dP_raw
            )

            # Model 4: Duration & Upper/Lower Mixer Aware Huber
            w_m4 = MODEL_WEIGHTS["model4_duration"]
            d_Ti = f["T_active"] - base_f["T_active"]
            d_Pup = f["P_up"] - base_f["P_up"]
            d_Plow = f["P_low"] - base_f["P_low"]
            pred_m4 = (
                w_m4["b0"] +
                w_m4["w_Ti"] * d_Ti +
                w_m4["w_T_w01"] * dT_w01 +
                w_m4["w_P_up_mean"] * d_Pup +
                w_m4["w_P_low_mean"] * d_Plow +
                w_m4["w_Pmean"] * dP_active
            )

            # Recommended Consensus (Average of Active & Phase 4)
            pred_active = clean_float(pred_active, 0.0)
            pred_phase4 = clean_float(pred_phase4, 0.0)
            pred_raw = clean_float(pred_raw, 0.0)
            pred_m4 = clean_float(pred_m4, 0.0)
            pred_ensemble = clean_float(round(0.5 * pred_active + 0.5 * pred_phase4, 2), 0.0)

        # Check if this batch has a direct lab test
        actual_lab_val = None
        for lt in lab_tests:
            if lt["mapped_batch"] == b:
                actual_lab_val = lt["mooney_value"]
                break

        full_batch_results.append({
            "batch_number": b,
            "pallet_id": f["pallet_id"],
            "delta_mooney_pred": round(clean_float(pred_ensemble, 0.0), 2),
            "delta_mooney_active": round(clean_float(pred_active, 0.0), 2),
            "delta_mooney_phase4": round(clean_float(pred_phase4, 0.0), 2),
            "delta_mooney_raw": round(clean_float(pred_raw, 0.0), 2),
            "delta_mooney_m4": round(clean_float(pred_m4, 0.0), 2),
            "pred_abs_mooney": round(clean_float(base_lab_mooney + pred_ensemble, 0.0), 2) if base_lab_mooney is not None else None,
            "actual_lab_mooney": actual_lab_val,
            "T0": round(clean_float(f["T0"], 0.0), 1),
            "T_discharge": round(clean_float(f["T_discharge"], 0.0), 1),
            "P_active": round(clean_float(f["P_active"], 0.0), 1),
            "P_up": round(clean_float(f["P_up"], 0.0), 1),
            "P_low": round(clean_float(f["P_low"], 0.0), 1),
            "E_discharge": round(clean_float(f["E_discharge"], 0.0), 2),
            "T_active": int(clean_float(f["T_active"], 0)),
            "T_upper": int(clean_float(f["T_upper"], 0))
        })

    # Summary statistics
    all_deltas = [clean_float(r["delta_mooney_pred"], 0.0) for r in full_batch_results]
    first_last_delta = full_batch_results[-1]["delta_mooney_pred"] if full_batch_results else 0.0
    first_last_active = full_batch_results[-1]["delta_mooney_active"] if full_batch_results else 0.0
    first_last_phase4 = full_batch_results[-1]["delta_mooney_phase4"] if full_batch_results else 0.0
    first_last_raw = full_batch_results[-1]["delta_mooney_raw"] if full_batch_results else 0.0
    first_last_m4 = full_batch_results[-1]["delta_mooney_m4"] if full_batch_results else 0.0

    # Actual lab delta if available
    actual_first_last = None
    if base_lab_mooney is not None and full_batch_results[-1]["actual_lab_mooney"] is not None:
        actual_first_last = round(full_batch_results[-1]["actual_lab_mooney"] - base_lab_mooney, 2)

    result = {
        "order_id": order_id,
        "equipment_id": equipment_id,
        "compound_code": compound_code,
        "compound_name": compound_name,
        "total_batches": total_batches,
        "total_pallets": total_pallets,
        "start_time": start_time_str,
        "end_time": end_time_str,
        "baseline_batch": baseline_batch,
        "has_lab_tests": has_lab_tests,
        "lab_tests": lab_tests,
        "base_lab_mooney": base_lab_mooney,
        "kpis": {
            "first_last_delta": first_last_delta,
            "first_last_active": first_last_active,
            "first_last_phase4": first_last_phase4,
            "first_last_raw": first_last_raw,
            "first_last_m4": first_last_m4,
            "actual_first_last": actual_first_last,
            "min_delta": round(min(all_deltas), 2) if all_deltas else 0.0,
            "max_delta": round(max(all_deltas), 2) if all_deltas else 0.0,
            "delta_T0_first_last": round(batch_features[sorted_batches[-1]]["T0"] - base_f["T0"], 1),
            "delta_P_active_first_last": round(batch_features[sorted_batches[-1]]["P_active"] - base_f["P_active"], 1),
            "delta_E_first_last": round(batch_features[sorted_batches[-1]]["E_discharge"] - base_f["E_discharge"], 2)
        },
        "batch_list": sorted_batches,
        "full_batch_results": full_batch_results,
        "elapsed_seconds": round(time.time() - t_start, 2)
    }

    # Store in cache and persist curves cache
    _ORDER_CACHE[cache_key] = result
    _ORDER_CACHE[f"curves_{order_id}"] = batch_curves_cache
    _ORDER_CACHE[f"features_{order_id}"] = batch_features

    return result


def get_batch_comparison(order_id: str, batch_a: int = 1, batch_b: Optional[int] = None) -> Dict[str, Any]:
    """
    Generate deep physical comparison between any two batches:
    1. 11-Stage breakdown table with exact mechanism & contributions
    2. Synchronized time-series curves (Power, Temp, Energy, Ram, Speed, Deviations)
    3. Waterfall attribution components
    """
    order_id = str(order_id).strip()
    curves_key = f"curves_{order_id}"
    features_key = f"features_{order_id}"

    # Ensure order is loaded
    if curves_key not in _ORDER_CACHE or features_key not in _ORDER_CACHE:
        analyze_order_full(order_id)

    curves_cache = _ORDER_CACHE[curves_key]
    features_cache = _ORDER_CACHE[features_key]

    available_batches = sorted(curves_cache.keys())
    if not available_batches:
        raise ValueError(f"No batch curve data cached for order {order_id}")

    if batch_a not in curves_cache:
        batch_a = available_batches[0]
    if batch_b is None or batch_b not in curves_cache:
        batch_b = available_batches[-1]

    ca = curves_cache[batch_a]
    cb = curves_cache[batch_b]
    fa = features_cache[batch_a]
    fb = features_cache[batch_b]

    # Align time series (1 Hz)
    max_len = max(ca["L"], cb["L"])
    time_series = list(range(max_len))

    def pad_arr(arr, target_len):
        if len(arr) >= target_len:
            return arr[:target_len].tolist()
        res = arr.tolist()
        res.extend([None] * (target_len - len(arr)))
        return res

    p_a = pad_arr(ca["power"], max_len)
    p_b = pad_arr(cb["power"], max_len)
    t_a = pad_arr(ca["temp"], max_len)
    t_b = pad_arr(cb["temp"], max_len)
    e_a = pad_arr(ca["energy"], max_len)
    e_b = pad_arr(cb["energy"], max_len)
    s_a = pad_arr(ca["speed"], max_len)
    s_b = pad_arr(cb["speed"], max_len)
    r_a = pad_arr(ca["ram"], max_len)
    r_b = pad_arr(cb["ram"], max_len)

    delta_p = []
    delta_t = []
    for i in range(max_len):
        if p_a[i] is not None and p_b[i] is not None:
            delta_p.append(round(p_b[i] - p_a[i], 1))
        else:
            delta_p.append(None)

        if t_a[i] is not None and t_b[i] is not None:
            delta_t.append(round(t_b[i] - t_a[i], 1))
        else:
            delta_t.append(None)

    # 11-Stage breakdown table
    table_rows = []
    for defn in STEP_DEFINITIONS:
        st = defn["step"]
        w_idx = st - 1
        w_key = f"w{w_idx:02d}"

        t1 = round(fa.get(f"T_{w_key}", 0.0), 1)
        t2 = round(fb.get(f"T_{w_key}", 0.0), 1)
        dt = round(t2 - t1, 1)

        p1 = round(fa.get(f"P_{w_key}", 0.0), 1)
        p2 = round(fb.get(f"P_{w_key}", 0.0), 1)
        dp = round(p2 - p1, 1)

        s1 = round(fa.get(f"S_{w_key}", 0.0), 1)
        s2 = round(fb.get(f"S_{w_key}", 0.0), 1)
        ds = round(s2 - s1, 1)

        r1 = round(fa.get(f"R_{w_key}", 0.0), 0)
        r2 = round(fb.get(f"R_{w_key}", 0.0), 0)

        # Stage specific contribution to Mooney
        contrib_mu = 0.0
        if st == 2:  # T_w01
            contrib_mu = round(dt * MODEL_WEIGHTS["active_huber"]["w_T_w01"], 2)
        elif st == 6:  # P_w05
            contrib_mu = round(dp * MODEL_WEIGHTS["active_huber"]["w_P_w05"], 2)
        elif st == 7:  # P_w06
            contrib_mu = round(dp * MODEL_WEIGHTS["active_huber"]["w_P_w06"], 2)

        table_rows.append({
            "step": st,
            "window": defn["window"],
            "name": defn["name"],
            "desc": defn["desc"],
            "T1": t1, "T2": t2, "dT": dt,
            "P1": p1, "P2": p2, "dP": dp,
            "S1": s1, "S2": s2, "dS": ds,
            "R1": r1, "R2": r2,
            "mooney_contrib": contrib_mu
        })

    # Waterfall attribution
    w_act = MODEL_WEIGHTS["active_huber"]
    dT_w01 = fb["T_w01"] - fa["T_w01"]
    dP_w05 = fb["P_w05"] - fa["P_w05"]
    dT_w05 = fb["T_w05"] - fa["T_w05"]
    dP_w06 = fb["P_w06"] - fa["P_w06"]
    dP_active = fb["P_active"] - fa["P_active"]

    c_b0 = w_act["b0"]
    c_tw01 = round(dT_w01 * w_act["w_T_w01"], 2)
    c_pw05 = round(dP_w05 * w_act["w_P_w05"], 2)
    c_tw05 = round(dT_w05 * w_act["w_T_w05"], 2)
    c_pw06 = round(dP_w06 * w_act["w_P_w06"], 2)
    c_pact = round(dP_active * w_act["w_P_active"], 2)
    pred_total = round(c_b0 + c_tw01 + c_pw05 + c_tw05 + c_pw06 + c_pact, 2)

    waterfall_data = [
        {"name": "模型基准截距", "value": c_b0, "type": "intercept"},
        {"name": "一段初温影响 (ΔT_w01)", "value": c_tw01, "type": "positive" if c_tw01 >= 0 else "negative"},
        {"name": "二段捏合功率 (ΔP_w05)", "value": c_pw05, "type": "positive" if c_pw05 >= 0 else "negative"},
        {"name": "控温段温度 (ΔT_w05)", "value": c_tw05, "type": "positive" if c_tw05 >= 0 else "negative"},
        {"name": "上密出料剪切 (ΔP_w06)", "value": c_pw06, "type": "positive" if c_pw06 >= 0 else "negative"},
        {"name": "有效总剪切功 (ΔP_act)", "value": c_pact, "type": "positive" if c_pact >= 0 else "negative"},
        {"name": f"预测车次差 (ΔMooney B{batch_b}-B{batch_a})", "value": pred_total, "type": "total"}
    ]

    return {
        "order_id": order_id,
        "batch_a": batch_a,
        "batch_b": batch_b,
        "meta_a": {
            "T0": round(fa["T0"], 1),
            "P_active": round(fa["P_active"], 1),
            "P_up": round(fa["P_up"], 1),
            "P_low": round(fa["P_low"], 1),
            "E_tot": round(fa["E_discharge"], 2),
            "T_active": fa["T_active"],
            "T_upper": fa["T_upper"]
        },
        "meta_b": {
            "T0": round(fb["T0"], 1),
            "P_active": round(fb["P_active"], 1),
            "P_up": round(fb["P_up"], 1),
            "P_low": round(fb["P_low"], 1),
            "E_tot": round(fb["E_discharge"], 2),
            "T_active": fb["T_active"],
            "T_upper": fb["T_upper"]
        },
        "table_rows": table_rows,
        "waterfall": waterfall_data,
        "pred_total_mu": pred_total,
        "curves": {
            "time": time_series,
            "power_a": p_a,
            "power_b": p_b,
            "delta_p": delta_p,
            "temp_a": t_a,
            "temp_b": t_b,
            "delta_t": delta_t,
            "energy_a": e_a,
            "energy_b": e_b,
            "speed_a": s_a,
            "speed_b": s_b,
            "ram_a": r_a,
            "ram_b": r_b
        }
    }
