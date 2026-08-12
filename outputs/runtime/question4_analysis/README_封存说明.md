# 问题四封存说明

## 封存状态

问题四已于 2026-08-10 封存。完整自动化回归验证：`99 passed`。

## 题目要求与交付对应

| 题目要求 | 封存交付 |
|---|---|
| 大型文旅活动情景 | `q4_event_counterfactual_2026.csv` |
| 持续性阴雨情景 | `q4_rain_counterfactual_2026.csv` |
| 基线适配与资源再优化 | `q4_baseline_reoptimization_summary_2026.csv`、`q4_scenario_window_adjustment_summary.csv` |
| 参数鲁棒性 | `q4_robustness_sensitivity_ranking.csv` |
| 成本—体验权衡参数 | `q4_lambda_tradeoff_2026.csv` |
| 论文图 | `q4_scenario_service_gap.png`、`q4_sensitivity_ranking.png` |
| 管理建议、局限与写作口径 | `论文写作/问题四_建模与写作说明.md` |

## 固定口径

- 两类情景均为反事实规划模拟，不是对真实活动客流或实际降雨影响的统计复原。
- 运营投入与体验损失为相对情景指标，不是人民币成本或问卷满意度。
- 泊位、车辆和人员均为规划推荐配置；仅公开核验的设施节点可称为已知容量。

## 后续修改规则

若新增真实活动预售、停车台账、接驳调度日志或小时级气象数据，应复制本封存版本后另建校准分支；不得直接覆盖本目录中的封存结果。
