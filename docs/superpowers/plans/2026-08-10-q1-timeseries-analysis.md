# Question 1 Time-series Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce auditable outputs for every descriptive and explanatory requirement of Question 1.

**Architecture:** Pure analysis functions transform official annual totals, continuous regional relative pressure, fitted joint-model coefficients and time-window evidence into CSV tables. A thin script reads project inputs and writes output artifacts.

**Tech Stack:** Python, pandas, numpy, pytest.

---

### Task 1: Implement annual and regional time-series analysis

**Files:**
- Create: `src/analysis/q1_timeseries.py`
- Create: `tests/test_q1_timeseries.py`

- [ ] Write failing tests for annual residual identity, three-part regional decomposition reconstruction, and complete season labels.
- [ ] Run `python -m pytest tests/test_q1_timeseries.py -q -p no:cacheprovider` and confirm the module-import failure.
- [ ] Implement `analyze_city_annual_trend`, `decompose_regional_pressure`, and `classify_regional_seasons` with required-column validation.
- [ ] Re-run the focused tests and require all pass.

### Task 2: Implement factor and service-time evidence summaries

**Files:**
- Modify: `src/analysis/q1_timeseries.py`
- Modify: `tests/test_q1_timeseries.py`

- [ ] Write failing tests requiring factor ranking by absolute standardized coefficient and service records to preserve `not_observed_hourly_flow=True`.
- [ ] Run the focused tests and confirm the missing-function failure.
- [ ] Implement `rank_factor_strength` and `build_service_time_scenarios`.
- [ ] Re-run the focused tests and require all pass.

### Task 3: Build reproducible Question 1 analysis artifacts

**Files:**
- Create: `scripts/build_q1_timeseries_analysis.py`
- Modify: `tests/test_q1_timeseries.py`

- [ ] Write a failing artifact-layout test against a temporary output directory.
- [ ] Implement `build_q1_analysis_outputs` and a script entry point that writes annual, seasonal, decomposition, factor, service scenario and quality-report files.
- [ ] Run the focused tests and require all pass.

### Task 4: Package the completed Question 1 handoff

**Files:**
- Modify: `问题一_模型说明与封装.md`
- Modify: `data/model_input/CSV文件作用说明.md`

- [ ] Update model handoff with exact Question 1 requirement-to-output mapping and all estimation limits.
- [ ] Update CSV guide to identify joint parameter diagnostics and Question 1 analysis outputs.
- [ ] Run the entire test suite and run both model rebuild scripts plus the Question 1 analysis script.
- [ ] Inspect row counts, required labels and quality report before claiming Question 1 is ready to seal.
