# 问题一：排名截断观测层设计

## 目标

在不虚构未上榜景点客流的前提下，将地图客流排名数据作为“已观测密度 + 未观测左删失”纳入秦皇岛三大片区连续日游客规模估计。

## 范围与非目标

- 覆盖 `censored_attraction_observation_panel.csv` 中的实际采样日及 BDH、HGA、SHG 景点记录。
- 不把未采样日放入删失观测层；连续日生成仍由现有日历、天气、滞后搜索特征的压力基线完成。
- 不把 `left_censored` 记录置零，也不为单个未上榜景点生成伪造的绝对游客数。
- 不改写 CITY、ARA、BND 等与三大片区不同口径的锚点。

## 数据流

1. 读取景点—采样日面板。`density` 行提供相对客流指数；`left_censored` 行仅表示同片区同采样日未进入原始排名观测。
2. 对每个 `date × region_code` 计算观测阈值 `c_{r,t}`：该组已观测 `visitor_index` 的最小正值。若该组没有密度观测，则不构造阈值或似然贡献。
3. 输出观测层诊断表，包含：密度观测数、左删失数、删失比例、阈值、观测对数指数均值/标准差，以及正态左删失近似下的删失概率和对数似然贡献。
4. 将诊断表以 `date × region_code` 合并到连续日压力基线的采样日期，用于评估与校正选择偏差；连续日压力和锚点尺度估计仍保留其已有标签与不确定性情景。

## 统计定义

令景点相对指数对数为 `z_{i,t}=log(1+visitor_index_{i,t})`，片区采样日的基准均值和标准差为 `mu_{r,t}`、`sigma_{r,t}`。

- 对 `density` 行，保留其对数密度贡献 `log f(z_{i,t}|mu_{r,t},sigma_{r,t})`。
- 对 `left_censored` 行，只加入 `log P(z_{i,t} <= log(1+c_{r,t}))`。
- 当样本量不足以稳定估计 `sigma` 时，使用预声明的最小标准差下限，并标记 `uncertainty_flag=small_observed_sample`。

该层是对排名选择机制的近似校正：它不声称知道未上榜景点的真实指数，也不把诊断似然解释为绝对客流量。

## 输出

新增 `data/model_input/censored_observation_diagnostics.csv`：

- `date`, `region_code`, `density_count`, `left_censored_count`, `censor_rate`
- `censor_threshold_index`, `log_index_mean`, `log_index_std`, `uncertainty_flag`
- `density_log_likelihood`, `censor_log_likelihood`, `total_log_likelihood`

新增质量报告，记录可用采样日数、无阈值组数、小样本组数和删失比例分布。

## 校验标准

- 未采样日期不得出现在诊断表中。
- 每个有效 `date × region_code` 的 `density_count + left_censored_count` 必须等于面板对应行数。
- `left_censored_count=0` 时删失对数似然应为 0。
- 只有左删失、没有密度观测的组必须明确标为不可估计，而非补充阈值。
- 原有连续日压力序列、锚点台账和尺度估计的标签、区域口径不得被改写。

## 风险与解释边界

原始排名采样不连续、覆盖范围有限；因此该层仅处理“同一实际采样日内的未上榜”选择机制，不将其外推为每日完整景点普查。问题一最终日游客规模仍应标注为锚点约束估计，并单列锚点冲突和跨年外推情景。
