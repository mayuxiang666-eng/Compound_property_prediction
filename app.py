"""
Continental Final Mix Mooney Intelligence Platform
FastAPI Application serving the interactive dashboard and real-time analysis APIs.
"""

import os
import sys
import io
import csv
import math
import numpy as np
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
from typing import Optional, Any
from fastapi import FastAPI, Query, HTTPException, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from engine import mooney_prediction_engine as engine


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace all NaN and Inf values with JSON-compliant safe values."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, (np.floating, np.integer)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    return obj

app = FastAPI(
    title="终胶密炼过程曲线与门尼全车次智能预测平台",
    description="Continental Final Mix Mooney Intelligence Platform",
    version="2.0.0"
)

# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

CURATED_ORDERS = [
    {
        "order_id": "7045560",
        "tag": "最新未化验盲测工单",
        "title": "7045560 (2026-10-27 77车 ML08)",
        "desc": "机台热平衡饱和，初温升幅 +37.4°C，有效混炼剪切功率塌缩 -31.9 kW，净做功减少 -2.0 kWh。"
    },
    {
        "order_id": "8176639",
        "tag": "80车实测对验基准",
        "title": "8176639 (80车经典工单 3点化验)",
        "desc": "首车 67.20 -> 尾车 69.17，实测增幅 +1.97 MU，验证前向物理预测与化验真值高契合度。"
    },
    {
        "order_id": "8149782",
        "tag": "大门尼波动工单",
        "title": "8149782 (高门尼偏差实测)",
        "desc": "首车 72.45 -> 尾车 74.60，实际波动 +2.15 MU，验证中高门尼区间的剪切功率响应。"
    },
    {
        "order_id": "7894359",
        "tag": "83车连续生产周期",
        "title": "7894359 (83车大批量连续生产)",
        "desc": "全周期连续混炼，首车 73.47 -> 尾车 72.84，全生命周期波动预测与阶段特征追踪。"
    },
    {
        "order_id": "8197867",
        "tag": "双样本托盘防错验证",
        "title": "8197867 (双样本托盘映射)",
        "desc": "化验室仅检测 2 个样本（001与017），第2样本为第3托盘（车次9~11），验证托盘防错算法。"
    }
]


@app.get("/", response_class=HTMLResponse)
async def serve_index():
    """Serve the single-page dashboard HTML."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Dashboard template not found")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.get("/api/curated_orders")
async def get_curated_orders():
    """Get list of pre-configured sample orders."""
    return JSONResponse(content={"orders": CURATED_ORDERS})


@app.get("/api/analyze")
async def analyze_order(
    order_id: str = Query(..., description="SFE Order ID"),
    baseline: int = Query(1, description="Baseline batch number for delta"),
    refresh: bool = Query(False, description="Force database re-query")
):
    """
    Perform full order analysis:
    - Real-time query BatchHeader, BatchCurve, Lab Tests
    - Physical feature extraction for all batches
    - Mooney delta prediction for all batches
    """
    try:
        res = engine.analyze_order_full(order_id=order_id, baseline_batch=baseline, force_refresh=refresh)
        return JSONResponse(content=sanitize_for_json(res))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "order_id": order_id})


@app.get("/api/batch_comparison")
async def batch_comparison(
    order_id: str = Query(..., description="SFE Order ID"),
    batch_a: int = Query(1, description="Baseline Batch A"),
    batch_b: Optional[int] = Query(None, description="Target Batch B")
):
    """
    Get detailed physical comparison between Batch A and Batch B:
    - 11-Stage table
    - Synchronized time series curves
    - Waterfall attribution breakdown
    """
    try:
        res = engine.get_batch_comparison(order_id=order_id, batch_a=batch_a, batch_b=batch_b)
        return JSONResponse(content=sanitize_for_json(res))
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e), "order_id": order_id})


@app.get("/api/export_csv")
async def export_csv(order_id: str = Query(..., description="SFE Order ID")):
    """Export full batch metrics and predictions as CSV."""
    try:
        res = engine.analyze_order_full(order_id=order_id)
        batch_rows = res.get("full_batch_results", [])
        if not batch_rows:
            raise HTTPException(status_code=404, detail="No batch data to export")

        output = io.StringIO()
        fieldnames = [
            "batch_number", "pallet_id", "delta_mooney_pred", "delta_mooney_active",
            "delta_mooney_phase4", "delta_mooney_raw", "pred_abs_mooney", "actual_lab_mooney",
            "T0", "T_discharge", "P_active", "P_up", "P_low", "E_discharge", "T_active", "T_upper"
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in batch_rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

        output.seek(0)
        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=Order_{order_id}_Mooney_Analysis.csv"}
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    import uvicorn
    print("======================================================================")
    print(" Continental Final Mix Mooney Intelligence Platform is starting...")
    print(" Local URL: http://127.0.0.1:5050")
    print("======================================================================")
    uvicorn.run("app:app", host="127.0.0.1", port=5050, reload=False, log_level="info")
