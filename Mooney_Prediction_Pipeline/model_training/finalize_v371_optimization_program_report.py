"""Build the evidence-backed V3.7.1 and dual-target optimization report."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports" / "v371_challenger" / "global_rds_active_v1"
OUT = BASE / "v371_optimization_program_review"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalized_metrics(path: Path, model: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [column.lower() for column in frame.columns]
    frame["model"] = model
    return frame


def select_model_rows(frame: pd.DataFrame, candidate: str) -> pd.DataFrame:
    scopes = {"global", "T15760", "T15760_M1_06_002_PV46", "B00458"}
    selected = frame.loc[
        frame["candidate"].eq(candidate)
        & frame["split"].isin(["val", "oot"])
        & frame["scope"].isin(scopes)
    ].copy()
    return selected[
        [
            "model", "candidate", "split", "scope", "groups", "mae", "rmse",
            "bias", "r2", "actual_std", "prediction_std",
            "prediction_to_actual_std_ratio", "within_5_mny_rate",
        ]
    ]


def c3_rows() -> pd.DataFrame:
    frame = pd.read_csv(BASE / "c3_detailed_metrics_and_rolling_baseline.csv")
    frame = frame.loc[frame["predictor"].eq("C3Prediction")].copy()
    parts = frame["slice"].str.rsplit("_", n=1, expand=True)
    frame["scope"] = parts[0].replace({"global": "global"})
    frame["split"] = parts[1]
    frame = frame.loc[frame["scope"].isin(["global", "T15760", "B00458"])]
    frame["model"] = "C3"
    frame["candidate"] = "C3"
    frame["groups"] = frame["n"]
    frame["prediction_std"] = frame["pred_std"]
    frame["prediction_to_actual_std_ratio"] = frame["std_ratio"]
    frame["within_5_mny_rate"] = frame["hit_pm5"]
    frame.columns = [column.lower() for column in frame.columns]
    return frame[
        [
            "model", "candidate", "split", "scope", "groups", "mae", "rmse",
            "bias", "r2", "actual_std", "prediction_std",
            "prediction_to_actual_std_ratio", "within_5_mny_rate",
        ]
    ]


def fmt(value: float | int) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def table_lines(frame: pd.DataFrame, split: str) -> str:
    rows = frame.loc[
        frame["split"].eq(split) & frame["scope"].isin(["global", "T15760", "B00458"])
    ]
    lines = [
        "| 模型 | 切片 | Lab组 | MAE | RMSE | Bias | R² | 预测Std/实际Std |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in rows.iterrows():
        lines.append(
            f"| {row['model']} | {row['scope']} | {int(row['groups'])} | "
            f"{fmt(row['mae'])} | {fmt(row['rmse'])} | {fmt(row['bias'])} | "
            f"{fmt(row['r2'])} | {fmt(row['prediction_to_actual_std_ratio'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r1 = normalized_metrics(BASE / "v371_r1_temporal_group" / "r1_metrics.csv", "R1")
    r2 = normalized_metrics(BASE / "v371_r2_hierarchical_nominal" / "r2_metrics.csv", "R2")
    r1b = normalized_metrics(BASE / "v371_r1b_group_atomic_batch" / "r1b_metrics.csv", "R1B")
    r3 = normalized_metrics(BASE / "v371_r3_expert_ablation" / "r3_metrics.csv", "R3")
    comparison = pd.concat(
        [
            c3_rows(),
            select_model_rows(r1, "R1_REGRESSION"),
            select_model_rows(r2, "R2_HIERARCHICAL_NOMINAL"),
            select_model_rows(r1b, "R1B_GROUP_ATOMIC_BATCH"),
            select_model_rows(r3, "R3_SILICA_CB_K20"),
        ],
        ignore_index=True,
    )
    comparison.to_csv(OUT / "model_comparison_key_slices.csv", index=False)

    safe = pd.read_csv(
        BASE / "v371_safe_group_state_diagnostic" / "safe_group_state_metrics.csv"
    )
    safe_key = safe.loc[
        safe["history_groups"].isin([0, 5])
        & safe["split"].isin(["val", "oot"])
        & safe["scope"].isin(["global", "T15760"])
    ].copy()
    safe_key.to_csv(OUT / "safe_group_state_key_slices.csv", index=False)

    r3_summary = read_json(BASE / "v371_r3_expert_ablation" / "r3_summary.json")
    readiness = pd.read_csv(BASE / "dual_target_readiness" / "dual_target_readiness_matrix.csv")
    energy = pd.read_csv(
        ROOT / "V4_1_Energy_Optimization" / "reports" / "v41_energy_optimization"
        / "energy_prediction_metrics.csv"
    )
    b_response = read_json(
        ROOT / "reports" / "corrected_phase_history" / "active_exact_pilot_step_history_v1"
        / "b00458_main_template_process_response_v1" / "summary.json"
    )
    t_response = read_json(
        ROOT / "reports" / "recent_oot_t15760" / "process_response_validation" / "summary.json"
    )

    b_endpoint_rows = []
    for endpoint, values in b_response["endpoint_metrics"].items():
        b_endpoint_rows.append(
            {
                "pilot": "B00458",
                "endpoint": endpoint,
                "oot_samples": values["oot_samples"],
                "oot_mae": values["oot_mae"],
                "interval_coverage": values["oot_interval_coverage"],
                "passed": values["passed"],
            }
        )
    pd.DataFrame(b_endpoint_rows).to_csv(
        OUT / "b00458_process_response_key_metrics.csv", index=False
    )

    report = f"""# V3.7.1 优化与双目标 Step 推荐综合实验报告

生成时间：{datetime.now(timezone.utc).isoformat()}

## 一、结论

本轮完成了 V3.7.1 数据/指标合同修正、四阶段专家架构的有限优化、组级安全状态回放、严格 Step→实际 Phase 合同和门尼+能耗双目标优化器。当前最优开发候选是 `{r3_summary['selected_candidate']}`，四专家结构应保留，但**尚不具备影子推荐放行资格**；四条材料体系×路线的双目标就绪结果为 `{int(readiness['dual_target_ready'].sum())}/{len(readiness)}`。

这不是继续扩大超参数搜索就能解决的问题。当前瓶颈依次是：新的未查看时间盲窗、完整实际 Phase 响应、当前 ProcessVersion/Template 绑定，以及能耗模型正式影子排序批准。

## 二、这次修正的关键设计缺陷

1. **Lab 粒度错误风险**：一个 Lab MNY 对应多个车次。正式指标已固定为“逐车预测→按 `OrderID+PalletID` 求预测均值→一个 Lab 组只比较一次”，不再把同一 Lab 重复 N 次计权。
2. **随机交叉验证与组权重**：旧候选包含随机 KFold/配方均衡权重，不能代表实际时间外推。本轮使用按 Lab 时间扩展窗，每个 Lab 组总权重恒为 1。
3. **Stage1b/Stage2 训练内泄漏**：现在只用 temporal OOF Stage1 与 prior-only Stage1b 残差训练 Stage2。
4. **Stage3 指标虚高**：生产完成时只能用已返回的独立历史 Lab 组。安全回放显示过去 5 组 Offset 在验证期有收益、在 T15760 OOT 反而恶化，因此 Stage3 继续关闭。
5. **组均值模型可能损失车次信息**：R1B 已用批次行训练、每 Lab 组有效权重为 1，再对批次预测求组均值；结果未显著改善，证明当前误差并非由评价公式算错造成。
6. **模板 nominal 过细**：R2 的 OOT 只有 12 组命中精确 Template，202 组退化到 Recipe+Mixer，154 组退化到 Material+Route，导致稀疏回退和泛化恶化。
7. **Endpoint 冒充 Phase 特征**：推荐链现在只接受 `v37_actual__<feature>__p50`；五个阶段终点不能再直接替代完整 V3.7 Phase 输入。
8. **旧 OOT 已被多轮查看**：它现在只能用于开发比较，不能继续作为生产放行证据。

## 三、模型实验结果

### 验证集

{table_lines(comparison, 'val')}

### 已查看 OOT（仅开发诊断）

{table_lines(comparison, 'oot')}

解释：R3 的验证集 global MAE 最好，为 3.893；T15760 验证 MAE 5.257，Bias 已接近 1，但 R² 仅约 0.008。OOT T15760 MAE 4.914、R² 仍为负，B00458 只有 15 个组且 OOT R² 为负。因此不能用 global 指标掩盖试点配方波动学习不足。

历史 V3.7 报告中的约 2.5 MAE 不能与本表直接等价比较：其样本行、Stage3 状态、随机切分及重复 Lab 标签权重不同。当前合同下，C3 与 R1B 的对照已经排除了“仅仅把 Lab 比较方式算错”这一解释。

## 四、四阶段架构是否保留

保留。R3 有限消融显示，同时保留 Silica 的 PID/Wet/Bottom/Material 子专家与 CarbonBlack 的 Preparation/Bottom/Material 子专家，并把 Stage1b shrinkage 调到 20，是当前最稳的开发候选。后续不做无边界调参，优先补信号与独立验证。

正式结构仍为：

```text
Stage1 配方静态表面
→ Stage1b 配方/配方族偏置收缩（时间交叉拟合）
→ Stage2 材料体系×路线专家及子专家过程残差
→ Stage3 关闭
```

## 五、过程响应和双目标优化现状

- B00458 精确配方 `{b_response['recipe_id']}`：{b_response['history_batches']} 批/{b_response['orders']} 单。排胶温度 MAE 1.343°C 通过，但 Dry、油后和 Wet 终点分别约 3.207/2.822/3.683°C；Wet duration 覆盖率 79.3% 未达 80%，所以整体 `CONTROL_TRANSFER_UNAVAILABLE`。
- T15760：主模板 {t_response['dominant_template_batches']} 批、占比 {100*t_response['dominant_template_share']:.1f}%，但全部必要输出未通过，不能生成推荐。
- 能耗 Mode B 有建模价值，OilWet 的 CarbonBlack/Silica 路线 R² 分别约 0.895/0.768；但项目策略仍为 `NOT_RANKED`，且缺少同 TemplateKey 新盲窗和保守能耗上界，因此优化器必须阻断。
- 双目标优化器已实现 `QUALITY_FIRST` 与 `BALANCED_TARGET`。所有质量、安全、热预算、执行能力、OOD、模型路径和版本门禁先通过，才按目标门尼、目标 kWh、不确定性和动作成本排序。

## 六、当前生产配方的业务入口

首版不是车内实时控制，而是：

```text
用户输入 RecipeID + 目标门尼 + 目标kWh
→ 读取 RDS Active 当前 Order/ProcessVersion/Template/完整 Step Target 向量
→ 最近完成车次曲线完整且当前候选车相关 Step 尚未开始
→ 若版本/模板变化，旧车上下文自动失效但不否定新版本本身
→ 三模型链门禁
→ 完整 Step Target 向量候选
→ MANUAL_REVIEW_ONLY
```

当前请求合同会拒绝：本地 MMS 来源、已开始车次、未来/当前实际曲线特征、TargetVectorHash 错配、无完整前车及用户目标不在门尼规格内。工程师正常修改 ProcessVersion/Target 后会形成新的当前模板，不会被当成数据错误；但该模板必须重新满足精确或批准 pooled 模型门槛。

## 七、下一轮最关键实验

1. 冻结当前 R3 代码与特征合同，取得新的、从未查看的 Lab 时间窗；若没有新盲窗，不能声称重新训练后已放行。
2. 对 T15760/B00458 训练“完整 Step Target 向量→所有受影响实际 Phase 特征”的多输出模型；不能只做五个 endpoint。
3. 过程响应按当前 ProcessVersion/TemplateKey 验证 P10/P50/P90；B00458 优先解决 Dry/油后/Wet 温度响应，T15760优先解决 Stage6 与非单调 overreaction。
4. 能耗 Mode B 在四路线和当前模板上做相同时间盲窗、区间覆盖及高能耗 P90 子集验证；通过后才把 `NOT_RANKED` 改为 `SHADOW_RANKING_APPROVED`。
5. 三链同时通过后，先回放不少于 30 个独立 Lab 组，再生成下一车 `MANUAL_REVIEW_ONLY`，不写 Recipe/PLC。

## 八、安全与数据边界

- 本轮所有模型和审计读取冻结的 RDS Active 快照；数据库访问：否。
- 本地 MMS/Archive 访问：否；云端 Archive 状态固定为 `BLOCKED_ARCHIVE_NOT_REPLICATED`。
- Stage3：关闭；数据库写入：无；Recipe/PLC 写入：无；自动控制：关闭。
"""
    report_path = OUT / "V3.7.1优化与双目标推荐综合实验报告.md"
    report_path.write_text(report, encoding="utf-8")
    summary = {
        "contract": "V3.7.1-OPTIMIZATION-PROGRAM-REVIEW-V1",
        "selected_development_candidate": r3_summary["selected_candidate"],
        "mooney_release_status": r3_summary["release_status"],
        "dual_target_ready_routes": int(readiness["dual_target_ready"].sum()),
        "dual_target_total_routes": int(len(readiness)),
        "stage3_enabled": False,
        "automatic_control": False,
        "local_mms_access": False,
        "archive_status": "BLOCKED_ARCHIVE_NOT_REPLICATED",
        "report": str(report_path),
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
