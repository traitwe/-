# `data/model_input` CSV 文件作用说明

本目录存放 B 题由清洗数据进一步生成的模型输入、中间结果和可审计输出。除特别注明外，这些文件都可以由 `scripts/` 中的构建脚本重新生成；不要手工修改其中数值。

## 使用主链

问题一当前的主链为：

`search_keyword_rules.csv` → `daily_search_theme_features_2016_2026.csv`  
`censored_attraction_observation_panel.csv` + 日历/天气/搜索特征 → `daily_region_pressure_baseline_2023_2025.csv`  
`calibration_anchors_prepared.csv` → `calibration_anchor_scope_ledger.csv`  
压力基线 + 可直接标定锚点 → `daily_region_visitor_scale_estimates_2023_2025.csv`

## CSV 清单

| 文件 | 作用 | 核心字段/正确用法 | 当前定位 |
|---|---|---|---|
| `search_keyword_rules.csv` | 17 个百度指数关键词的人工规则表。 | `keyword`、质量等级 `quality_grade`、主题 `theme`；天气词仅作天气控制，不进入旅游需求主题。 | 搜索特征的规则输入。 |
| `daily_search_theme_features_2016_2026.csv` | 将 A/B 级关键词按年标准化后聚合成景点、海滨度假、目的地三类搜索主题，并生成滞后 1/2/3/7 天特征。 | `theme_*`、`theme_*_lag_1/2/3/7`；预测模型仅使用滞后项，避免使用未来搜索信息。问题二在真实观测日期内按片区选择目的地主题的一个候选滞后，选择记录另存于 `outputs/question2_analysis/q2_regional_search_lag_selection.csv`。 | 问题一压力基线的解释变量；问题二动态协变量。 |
| `q3_parameter_basis_register.csv` | 对相对资源优化中不可由本地运营台账直接标定的预算、相对成本、风险权重与安全系数逐项编号。 | `LA-R01`—`LA-R14`、参数值/规则、适用范围、依据类别与边界。 | 问题三、四的参数追溯附录；不将相对权重误写为人民币成本。 |
| `censored_attraction_observation_panel.csv` | 地图景点客流排名数据构造的“景点—实际采样日”面板。 | `density` 表示有相对客流指数观测；`left_censored` 表示同一采样日未上榜，绝不等于零客流；`visitor_index` 仅在 `density` 行有效。 | 问题一排名截断观测层的原始输入。 |
| `censored_observation_diagnostics.csv` | 将景点排名面板按采样日和片区汇总为左删失诊断。 | `density_count`、`left_censored_count`、`censor_threshold_index`、`total_log_likelihood`、`uncertainty_flag`；仅衡量同一实际采样日内的选择机制。 | 问题一左删失观测层输出；不含任何未上榜景点的补全客流。`no_positive_density_index` 表示只有零指数上榜记录，未强行构造阈值。 |
| `calibration_anchors_prepared.csv` | 将公开报道的游客量统一成可用于后续判断的期间总量锚点。 | `visitor_total`、`period_start/end`、`region_code`、`anchor_use`、`source_dataset`；CITY、BDH、HGA、SHG、ARA、BND 等口径不得直接相加。 | 锚点标准化中间表，保留全部可核验记录。 |
| `calibration_anchor_scope_ledger.csv` | 对标准化锚点做区域与统计范围审计。 | `scope_decision`：仅 `direct_region_scale` 能直接标定 BDH/HGA/SHG；`lower_bound_only` 仅作下界；`subarea_or_scenic_scope`、`outside_target_regions` 不混入片区总量。 | 问题一尺度标定的唯一锚点筛选依据。 |
| `daily_region_pressure_baseline_2023_2025.csv` | 日历两状态、天气、滞后搜索主题和区域效应拟合得到的三片区连续日相对压力。 | `pressure_index` 是相对指数；`estimate_label=relative_pressure_index`。 | 问题一的连续日相对客流形态；不能直接当游客人次或容量约束需求。 |
| `pressure_baseline_rolling_validation.csv` | 压力基线的历史前推验证结果。 | `train_end`、`test_rows`、`mae`；MAE 衡量相对地图指数误差。 | 比较问题一/问题二模型版本；不可解释为游客人数误差。 |
| `daily_region_visitor_scale_estimates_2023_2025.csv` | 由压力指数和通过口径审计的主锚点生成的 BDH/HGA/SHG 连续日游客规模情景。 | `visitor_estimate_conservative`、`visitor_estimate_baseline`、`visitor_estimate_high`；`scale_source_year` 和 `scale_interval_method` 标记直接标定或跨年外推。 | 问题一当前最重要的输出；用于问题二训练和问题三需求情景。所有数值均为锚点约束估计，不是观测真值。 |
| `visitor_scale_anchor_fit.csv` | 记录实际进入尺度计算的锚点回代情况。 | `calibration_role`：`primary_scale` 驱动尺度，`diagnostic_only` 仅报告冲突；`relative_fit_error` 衡量回代偏差。 | 论文中说明锚点一致性、冲突与不确定性的审计证据。 |
| `sampled_region_daily_pressure_baseline.csv` | 早期版本的“仅实际采样日”压力基线。 | 字段与连续日压力文件相近，但日期不连续。 | **历史保留，不再作为主链输入。** 请优先使用 `daily_region_pressure_baseline_2023_2025.csv`。 |

## 当前不要混用的概念

- `visitor_index`：地图相对指数，不是游客人次。
- `pressure_index`：模型得到的相对旅游压力，不是游客人次。
- `visitor_estimate_*`：由公开锚点约束得到的日游客规模**估计情景**，不是逐日真实统计值。
- CITY、BND、ARA 等锚点：可用于外部校验或扩展模型，但不能未经转换并入 BDH/HGA/SHG 三片区。

## 推荐读取顺序

1. 问题一主体数据：`daily_region_visitor_scale_estimates_2023_2025.csv`。
2. 检查不确定性与外推：同文件的 `scale_interval_method`，以及 `visitor_scale_anchor_fit.csv`。
3. 检查模型可靠性：`pressure_baseline_rolling_validation.csv`。
4. 研究排名观测机制：`censored_attraction_observation_panel.csv`；左删失诊断表将在后续问题一完善时新增。
