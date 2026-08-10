# B题新增数据预处理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将新增的承载标准、景区面积、片区面积情景、预约比例、日内服务证据和 17 关键词热度纳入可重复清洗层。

**Architecture:** 原始 CSV 保持不变。新增的资源参数表由专用清洗函数标准化为英语字段和数值上下界；17 关键词热度替代旧的 4 关键词建模输入，同时保留质量标记。清洗脚本在生成表后更新同一份质量报告。

**Tech Stack:** Python, pandas, direct-import tests.

---

### Task 1: 新增资源参数清洗函数

**Files:**
- Modify: `src/data/cleaning.py`
- Modify: `tests/test_cleaning.py`

- [ ] 写入失败测试：面积汇总必须输出片区、有效面积下限/基准/上限；承载标准必须把 `1-1.1` 拆为数值上下界；预约表只能保留明确可作为实际预约率的记录。
- [ ] 运行直接导入测试，确认函数尚不存在而失败。
- [ ] 实现 `clean_capacity_standard`、`clean_regional_area_scenarios`、`clean_reservation_scenarios` 和 `clean_service_time_evidence`；不填补缺失值。
- [ ] 重新运行新增测试并确认通过。

### Task 2: 接入清洗管线

**Files:**
- Modify: `scripts/clean_b_data.py`

- [ ] 读取 `L0_C20`、`L0_C21`、`L0_C21c`、`L0_C22` 和 `qinhuangdao_hourly_entry_time_evidence`。
- [ ] 生成 `capacity_space_standard.csv`、`attraction_effective_area.csv`、`regional_effective_area_scenarios.csv`、`reservation_ratio_scenarios.csv` 和 `hourly_service_time_evidence.csv`。
- [ ] 优先将 `L0_B5c` 的 17 关键词清洗为 `daily_search_heat_17_keywords_2016_2026.csv`，保留旧 4 关键词表以兼容已有分析。

### Task 3: 重建与核验

**Files:**
- Output: `data/clean/*`

- [ ] 运行全部清洗测试。
- [ ] 重建清洗层。
- [ ] 断言新增 6 张表存在、行数大于零、容量/面积边界数值有效，且 17 关键词热度的关键词数为 17。
- [ ] 读取质量报告，记录异常和不可直接用于秦皇岛实际预约率的记录。
