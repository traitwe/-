# B题模型输入

## `censored_attraction_observation_panel.csv`

由地图景点客流指数构造的“景点—日期”面板。面板只覆盖原始排名数据实际采样的日期；`density` 表示该采样日该景点进入了原始排名观测；`left_censored` 表示在同一采样日未进入观测，绝不等价于零客流。未采样日期不进入面板。`visitor_index` 只在 `density` 行有效，仍是相对指数。

## `censored_observation_diagnostics.csv`

按“采样日—片区”汇总排名截断观测的诊断表。它保留已观测指数的密度贡献，并把未上榜记录处理为低于该组观测阈值的左删失概率；不生成任何未上榜景点的客流数值。`uncertainty_flag` 标出无密度观测或小样本组。

## `calibration_anchors_prepared.csv`

公开报道的游客规模锚点。`visitor_total` 是按报道频率换算的对应期间总量，`anchor_use` 区分点标定与仅下界约束；`region_code` 和 `source_dataset` 保留口径与来源，CITY、BDH、SHG、HGA、ARA、BND 等不可在未说明的情况下相加或替换。

## `calibration_anchor_scope_ledger.csv`

锚点的口径审计台账。仅 `scope_decision=direct_region_scale` 的 BDH/HGA/SHG 记录可以进入片区日游客规模的直接标定；`lower_bound_only` 只用于不等式约束，子景区、城市总量及其他片区记录不与三片区总量混用。

## `b_model_input_quality_report.json`

记录输出行数、观测/左删失数量、日期范围、区域代码和锚点来源。每次重建模型输入后应先检查此文件。

## `daily_region_pressure_baseline_2023_2025.csv`

由日历两状态（常态/高压）、天气、滞后一天的搜索主题和区域固定效应拟合的第一版连续日相对压力指数，覆盖 BDH、SHG、HGA 在 2023—2025 年的日历日。`pressure_index` 是供后续锚点校准、层级模型初始化与比较的相对量；不得作为游客人次、也不得直接用于资源容量约束。

## `pressure_baseline_rolling_validation.csv`

以 2023-06-30 和 2024-06-30 为历史截点的前推验证结果。`mae` 衡量预测相对压力指数与后续实际地图客流指数之间的绝对误差，仅用于不同模型版本的比较，不能解释为“游客人数误差”。

## `daily_region_visitor_scale_estimates_2023_2025.csv`

三片区连续日游客规模情景估计。`visitor_estimate_baseline`、`visitor_estimate_conservative`、`visitor_estimate_high` 均由 `direct_region_scale` 锚点约束而来，仍是估计值而非观测值；`scale_source_year` 和 `scale_interval_method` 说明本年是直接标定还是沿用上一有效锚点。

## `visitor_scale_anchor_fit.csv`

每条实际参与标定的锚点对应的压力合计、隐含尺度、回代期间总量及相对拟合误差。它是论文中报告锚点一致性和不确定性的依据。

## 重建命令

```powershell
$py='C:\Users\lyy20\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py scripts\build_b_model_inputs.py
```
