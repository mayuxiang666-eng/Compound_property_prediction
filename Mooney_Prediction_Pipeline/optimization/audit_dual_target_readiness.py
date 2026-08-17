"""Generate current dual-target readiness matrix from frozen experiment artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Mooney_Prediction_Pipeline.optimization.dual_target_optimizer import (
    DualModelReadiness,
    DualTargetRequest,
    optimize_dual_target,
)


BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUTPUT = BASE / "dual_target_readiness"
ENERGY_DIR = ROOT / "V4_1_Energy_Optimization" / "reports" / "v41_energy_optimization"
B00458_RESPONSE = (
    ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
    / "b00458_main_template_process_response_v1" / "summary.json"
)
T15760_RESPONSE = ROOT / "reports" / "recent_oot_t15760" / "process_response_validation" / "summary.json"
T15760_STRICT = ROOT / "reports" / "recent_oot_t15760" / "strict_context_response_comparison" / "summary.json"


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for values in frame.itertuples(index=False, name=None):
        rendered = ["" if pd.isna(value) else str(value) for value in values]
        rows.append(rendered)
    return "\n".join(
        "| " + " | ".join(row) + " |"
        for row in [headers, ["---"] * len(headers), *rows]
    )


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_row() -> pd.DataFrame:
    return pd.DataFrame([
        {
            "candidate_id": "READINESS_PROBE_ONLY",
            "mooney_p10": 49.0,
            "mooney_p50": 50.0,
            "mooney_p90": 51.0,
            "energy_kwh_p10": 38.0,
            "energy_kwh_p50": 40.0,
            "energy_kwh_p90": 42.0,
            "model_path_status": "MODEL_PATH_OK",
            "strict_actual_phase_response_status": "ACTUAL_PHASE_RESPONSE_OK",
            "safety_status": "PASSED",
            "thermal_budget_status": "PASSED",
            "execution_capability_status": "PASSED",
            "ood_status": "IN_DISTRIBUTION",
            "action_cost": 10.0,
            "changed_step_count": 1,
            "model_response_delta": -1.0,
        }
    ])


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    r3 = load_json(BASE / "v371_r3_expert_ablation" / "r3_summary.json")
    r3_metrics = pd.read_csv(BASE / "v371_r3_expert_ablation" / "r3_metrics.csv")
    energy_metrics = pd.read_csv(ENERGY_DIR / "mode_b_branch_metrics.csv")
    b00458 = load_json(B00458_RESPONSE)
    t15760 = load_json(T15760_RESPONSE)
    t15760_strict = load_json(T15760_STRICT)
    policy = load_json(ROOT / "Mooney_Prediction_Pipeline" / "optimization" / "expert_policy_v1.json")

    process_status = {
        "CarbonBlack_OilWet": (
            "CONTROL_TRANSFER_APPROVED" if b00458.get("endpoint_response_approved")
            and b00458.get("formal_v37_control_transfer_status") == "ACTUAL_PHASE_RESPONSE_OK"
            else "CONTROL_TRANSFER_UNAVAILABLE"
        ),
        "Silica_OilWet": (
            "CONTROL_TRANSFER_APPROVED" if t15760.get("dominant_model_approved")
            and any(t15760_strict.get("variant_all_outputs_passed", {}).values())
            else "CONTROL_TRANSFER_UNAVAILABLE"
        ),
        "CarbonBlack_NoOilDry": "CONTROL_TRANSFER_NOT_VALIDATED",
        "Silica_NoOilDry": "CONTROL_TRANSFER_NOT_VALIDATED",
    }
    strict_status = {
        route: (
            "ACTUAL_PHASE_RESPONSE_OK"
            if status == "CONTROL_TRANSFER_APPROVED"
            else "BLOCKED_MISSING_ACTUAL_PHASE_FEATURE_RESPONSE"
        )
        for route, status in process_status.items()
    }

    rows = []
    blocked_audits = []
    for _, energy in energy_metrics.iterrows():
        route = str(energy["system_route_branch"])
        material, phase = route.split("_", maxsplit=1)
        pilot = "B00458" if route == "CarbonBlack_OilWet" else (
            "T15760" if route == "Silica_OilWet" else "NOT_SELECTED"
        )
        template_status = (
            "EXACT_OR_APPROVED_POOLED_TEMPLATE" if pilot in {"B00458", "T15760"}
            else "TEMPLATE_NOT_VALIDATED"
        )
        readiness = DualModelReadiness(
            mooney_model_status="NOT_RELEASED_GATES_FAILED",
            mooney_model_id=r3["selected_candidate"],
            stage3_enabled=False,
            process_response_status=process_status[route],
            strict_actual_phase_response_status=strict_status[route],
            energy_model_status=(
                "SHADOW_RANKING_APPROVED" if policy["energy_status"] == "RANKED"
                else "NOT_RANKED"
            ),
            energy_model_id="V4.1_MODE_B",
            energy_route_status=(
                "ROUTE_GATE_PASSED"
                if float(energy["R2"]) > 0.6 and float(energy["MAE_kWh"]) <= 2.5
                else "ROUTE_GATE_FAILED"
            ),
            template_status=template_status,
            recipe_version_status="CURRENT_VERSION_BINDING_NOT_EVALUATED",
        )
        blockers = readiness.blockers()
        rows.append(
            {
                "material_system": material,
                "phase_route": phase,
                "route": route,
                "pilot": pilot,
                "mooney_model": r3["selected_candidate"],
                "mooney_release_status": r3["release_status"],
                "process_response_status": process_status[route],
                "strict_actual_phase_response_status": strict_status[route],
                "energy_policy_status": policy["energy_status"],
                "energy_mode_b_mae_kwh": float(energy["MAE_kWh"]),
                "energy_mode_b_r2": float(energy["R2"]),
                "energy_route_status": readiness.energy_route_status,
                "template_status": template_status,
                "recipe_version_status": readiness.recipe_version_status,
                "dual_target_ready": len(blockers) == 0,
                "blockers": "|".join(blockers),
            }
        )
        request = DualTargetRequest(
            recipe_id=f"{pilot}_READINESS_PROBE",
            target_mooney=50.0,
            target_energy_kwh=40.0,
            mooney_spec_low=45.0,
            mooney_spec_high=55.0,
        )
        result = optimize_dual_target(candidate_row(), request, readiness)
        audit = result.audit_table.copy()
        audit["route"] = route
        audit["optimizer_status"] = result.status
        blocked_audits.append(audit)

    matrix = pd.DataFrame(rows).sort_values(["material_system", "phase_route"])
    blocked = pd.concat(blocked_audits, ignore_index=True)
    matrix.to_csv(OUTPUT / "dual_target_readiness_matrix.csv", index=False, encoding="utf-8-sig")
    blocked.to_csv(OUTPUT / "current_blocked_optimizer_audit.csv", index=False, encoding="utf-8-sig")

    selected_r3 = r3_metrics.loc[
        r3_metrics["candidate"].eq(r3["selected_candidate"])
        & r3_metrics["scope"].isin(["global", "T15760", "B00458"])
    ][["split", "scope", "groups", "mae", "rmse", "bias", "r2", "prediction_to_actual_std_ratio"]]
    report = f"""# 门尼 + 能耗双目标优化当前就绪评审

## 结论

统一双目标合同已实现并通过测试，但当前四条路线全部为 `dual_target_ready=false`，不能输出真实 `MANUAL_REVIEW_ONLY` Step 向量。阻断是主动安全结果：

1. 当前最优 raw 门尼候选 `{r3['selected_candidate']}` 未通过全局/T15760/B00458 放行门槛。
2. B00458 与 T15760 的 Step Target→实际 Phase 过程响应均未通过；终点温度不能替代 V3.7 Phase 均值/积分等输入。
3. 专家策略仍规定 `ENERGY_STATUS={policy['energy_status']}`；V4.1 Mode B 只能作为模型候选证据，尚不能正式参与最终排序。
4. 运行时必须绑定当前订单的 RecipeID、ProcessVersion、MixerLine、StepPattern、ControlLogic 和 TargetVector；本就绪审计没有假装绑定一个不存在的当前订单。

## 四路线矩阵

{markdown_table(matrix)}

## 当前最佳门尼候选指标

{markdown_table(selected_r3)}

## 过程响应证据

- B00458：186 批；排胶温度通过，但 Dry/油后/Wet 温度 MAE 超门槛，Wet duration 覆盖率不足；正式状态 `{b00458['formal_v37_control_transfer_status']}`。
- T15760：主模板 848 批、模板一致率约 90%，但 Target-only、Target+环境、Target+环境+上一批上下文三个版本均未全部通过。
- 新 V4 搜索入口已经强制 `v37_actual__<feature>__p50` 合同；旧 endpoint→Phase shortcut 不再具备推荐资格。

## 能耗证据

V4.1 Mode B 的路线指标可用于继续建模：CarbonBlack OilWet 和 Silica OilWet 的 R² 较高；Silica NoOilDry 明显不足。但在专家策略从 `NOT_RANKED` 改为经过正式批准的状态前，双目标优化器必须返回 `BLOCKED_ENERGY_MODEL_NOT_READY`。

## 接下来最关键的工作

1. 为 B00458/T15760 分别训练“完整 Step Target 向量→所有受影响实际 V3.7 Phase 特征”的响应模型，而不只预测五个终点。
2. 为 R3 门尼模型加入新的、从未查看的时间窗；当前 OOT 已被多轮实验查看，不能继续承担放行证据。
3. 能耗 Mode B 以相同 TemplateKey/当前版本做新时间窗验证，并建立 P10/P50/P90 或保守上界；高能耗 P90 子集必须单独通过。
4. 三条模型链都通过后，系统接受用户输入的 RecipeID、目标门尼和目标 kWh，在完整 Step 向量候选中先过质量/安全硬门，再按目标距离和能耗上界排序。

## 安全边界

- Active-only 冻结快照；本次数据库访问：`false`。
- 本地 MMS/Archive：`false`；Archive：`BLOCKED_ARCHIVE_NOT_REPLICATED`。
- Stage3、Recipe/PLC 写入、自动控制：全部关闭。
"""
    (OUTPUT / "双目标优化就绪评审.md").write_text(report, encoding="utf-8")
    summary = {
        "contract": "DUAL_TARGET_READINESS_V1",
        "routes": int(len(matrix)),
        "ready_routes": int(matrix["dual_target_ready"].sum()),
        "mooney_model": r3["selected_candidate"],
        "mooney_release_status": r3["release_status"],
        "energy_policy_status": policy["energy_status"],
        "stage3_enabled": False,
        "automatic_control": False,
        "database_access": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        **summary,
        "report": str(OUTPUT / "双目标优化就绪评审.md").encode("ascii", "backslashreplace").decode("ascii"),
    }, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
