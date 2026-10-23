"""
Mixing Curve Analysis System Client Module
Connects to the deployed Mixing Curve Analysis Web Service (http://10.246.97.155:9000).
Retrieves real-time/historical batch curve data, anomaly judgments (JudgeState, RamPositionJudgeState),
power curve anomaly alerts (干混/湿混/加油段能耗及功率偏移), and system alarms.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import urllib.parse

import pandas as pd
import requests

CONFIG_PATH = Path(__file__).resolve().parent / "db_config.json"
DEFAULT_BASE_URL = "http://10.246.97.155:9000"


def get_base_url() -> str:
    """Read the Mixing Curve System base URL from db_config.json or use default."""
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            return cfg.get("MixingCurveSystem", {}).get("base_url", DEFAULT_BASE_URL)
        except Exception:
            pass
    return DEFAULT_BASE_URL


def decode_hex_curve(hex_str: Optional[str]) -> List[int]:
    """
    Decode 4-character hex string into a list of 16-bit unsigned integers.
    Each 4 characters represent a sampling point (0.5s or 1.0s interval).
    """
    if not hex_str or not isinstance(hex_str, str):
        return []
    cleaned = hex_str.strip()
    return [int(cleaned[i:i+4], 16) for i in range(0, len(cleaned) - 3, 4)]


def get_batch_anomaly(
    order_id: Union[str, int], 
    batch_id: Union[str, int], 
    base_url: Optional[str] = None,
    timeout: int = 15
) -> Dict[str, Any]:
    """
    Query the Mixing Curve Analysis System for the specified order and batch.
    Extracts JudgeState, JudgeReason, RamPositionJudgeState, and calculates is_normal.

    Returns:
        dict containing:
            - found: bool
            - is_normal: bool (True if both JudgeState and RamPosition are Normal)
            - judge_state: str (e.g. 'Normal', 'FS:Module-Mark', 'MO:Module-Mark', 'EX:Experience-RubberSticking')
            - judge_reason: str (e.g. 'None', '类第一车异常', '经模型判定为异常', '该生产过程出现架胶')
            - ram_judge_state: str (e.g. 'Normal', 'Abnormal', '偏高', '偏低')
            - mixer: str (e.g. 'ML08', 'ML09', 'FM04')
            - comp_name: str
            - product_time: str
            - order_id: str
            - batch_id: str
            - raw_record: dict (complete record from system)
    """
    server = (base_url or get_base_url()).rstrip("/")
    url = f"{server}/api/CurveAnalysisHistory/MixingCurveData"
    params = {"orderId": str(order_id), "batchId": str(batch_id)}

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return {
            "found": False,
            "error": f"Failed to connect to Mixing Curve API: {exc}",
            "order_id": str(order_id),
            "batch_id": str(batch_id),
            "is_normal": None,
        }

    all_curves = data.get("allCurveData", [])
    target = None
    for c in all_curves:
        if str(c.get("OrderID")) == str(order_id) and str(c.get("BatchID")) == str(batch_id):
            target = c
            break

    if not target:
        # Fallback to the first record if OrderID matches
        for c in all_curves:
            if str(c.get("OrderID")) == str(order_id):
                target = c
                break

    if not target:
        return {
            "found": False,
            "order_id": str(order_id),
            "batch_id": str(batch_id),
            "is_normal": None,
            "message": f"Order {order_id} Batch {batch_id} not found in allCurveData ({len(all_curves)} peer records returned)",
        }

    judge_state = target.get("JudgeState", "Unknown")
    judge_reason = target.get("JudgeReason", "None")
    ram_state = target.get("RamPositionJudgeState", "")

    # Normal criteria: JudgeState is 'Normal' AND ram_state is 'Normal' or empty
    is_normal = (judge_state == "Normal") and (ram_state in ("Normal", "", None))

    return {
        "found": True,
        "is_normal": is_normal,
        "judge_state": judge_state,
        "judge_reason": judge_reason,
        "ram_judge_state": ram_state,
        "judge_mode": target.get("JudgeMode", "Auto"),
        "mixer": target.get("Mixer", ""),
        "comp_name": target.get("CompName", ""),
        "comp_class_name": target.get("CompClassName", ""),
        "product_time": target.get("ProductTime", ""),
        "order_id": str(target.get("OrderID", order_id)),
        "batch_id": str(target.get("BatchID", batch_id)),
        "temperature_curve": decode_hex_curve(target.get("TemperData")),
        "power_curve": decode_hex_curve(target.get("PowerData")),
        "ram_position_curve": decode_hex_curve(target.get("RamPositionData")),
        "raw_record": target,
    }


def get_power_anomalies(
    date_from: str, 
    date_to: str, 
    comp_name: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: int = 15
) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Query power curve anomalies (干混段能耗偏移, 油加料段峰值功率偏移, 湿混段功率波动偏移, etc.).
    If comp_name is provided, queries specific order details for that compound.
    """
    server = (base_url or get_base_url()).rstrip("/")
    if comp_name:
        url = f"{server}/api/SystemOverview/PowerFillerCompOrders"
        params = {"dateFrom": date_from, "dateTo": date_to, "compName": comp_name}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    else:
        url = f"{server}/api/SystemOverview/PowerFillerList"
        params = {"dateFrom": date_from, "dateTo": date_to}
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return pd.DataFrame(resp.json())


def get_recent_alarms(base_url: Optional[str] = None, timeout: int = 10) -> pd.DataFrame:
    """Fetch real-time active alarms across all production lines."""
    server = (base_url or get_base_url()).rstrip("/")
    url = f"{server}/api/SystemOverview/RecentAlarms"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


def get_anomaly_types(base_url: Optional[str] = None, timeout: int = 10) -> pd.DataFrame:
    """Fetch the standard mapping of anomaly types and descriptions."""
    server = (base_url or get_base_url()).rstrip("/")
    url = f"{server}/api/CurveAnalysisHistory/AnomalyTypes"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return pd.DataFrame(resp.json())


if __name__ == "__main__":
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("=== Testing Mixing Curve Analysis System API ===")
    base = get_base_url()
    print(f"Target Server: {base}")

    # 1. Fetch anomaly types
    print("\n--- 1. Anomaly Types Supported ---")
    df_types = get_anomaly_types()
    for _, row in df_types.iterrows():
        print(f"  [{row['abnormalType']}] -> {row['abnormalDescription']}")

    # 2. Check final mix 8184718-1
    print("\n--- 2. Final Mix Anomaly Check (8184718 Batch 1) ---")
    fm_res = get_batch_anomaly("8184718", 1)
    status_str = "正常 (Normal)" if fm_res.get("is_normal") else "异常 (Abnormal)"
    print(f"  车次: Order {fm_res.get('order_id')} Batch {fm_res.get('batch_id')}")
    print(f"  判定: {status_str}")
    print(f"  JudgeState: {fm_res.get('judge_state')} | 原因: {fm_res.get('judge_reason')}")
    print(f"  压栓状态: {fm_res.get('ram_judge_state')}")
    print(f"  曲线采样点数: 温度 {len(fm_res.get('temperature_curve', []))} 点, 功率 {len(fm_res.get('power_curve', []))} 点")

    # 3. Check mother batch 8185696-1
    print("\n--- 3. Masterbatch Anomaly Check (8185696 Batch 1) ---")
    mb_res = get_batch_anomaly("8185696", 1)
    status_str = "正常 (Normal)" if mb_res.get("is_normal") else "异常 (Abnormal)"
    print(f"  车次: Order {mb_res.get('order_id')} Batch {mb_res.get('batch_id')}")
    print(f"  判定: {status_str}")
    print(f"  JudgeState: {mb_res.get('judge_state')} | 原因: {mb_res.get('judge_reason')}")
    print(f"  压栓状态: {mb_res.get('ram_judge_state')}")

    # 4. Check power anomalies for T15760
    print("\n--- 4. Power Anomalies for M1-T15760 (Order 8185696) ---")
    comp_orders = get_power_anomalies("2026-10-18", "2026-10-22", comp_name="9922A1576030_PV27_002")
    flagged = [o for o in comp_orders.get("orders", []) if str(o.get("orderId")) == "8185696"]
    if flagged:
        fo = flagged[0]
        print(f"  功率判定: 【异常】")
        print(f"  报警详情: {fo.get('alertDetail')}")
    else:
        print("  功率判定: 无功率报警")
