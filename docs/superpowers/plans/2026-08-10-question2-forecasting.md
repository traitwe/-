# Question 2 Forecasting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce auditable citywide monthly and three-region daily tourism forecasts, with model comparison, independent validation, summer peak intervals, and weather-shock scenarios.

**Architecture:** Use a two-scale pipeline. A city annual model forecasts the official annual domestic-tourist total; a calendar-constrained monthly allocation converts that total into a 12-month estimate. A regional daily dynamic-regression model forecasts the existing anchor-constrained visitor-scale estimate and is assessed only where raw attraction observations exist. Calendar variables are deterministic; future search and weather are scenario inputs, so no future realised information enters a forecast.

**Tech Stack:** Python, NumPy, Pandas, Matplotlib, Pytest.

---

### Task 1: Implement reusable annual and monthly forecasting primitives

**Files:**
- Create: `src/models/question2_forecasting.py`
- Create: `tests/test_question2_forecasting.py`

- [ ] Write failing tests for annual rolling-origin splits, annual forecast intervals, and monthly shares that are non-negative and sum to one.
- [ ] Run the targeted tests and verify they fail because the module is absent.
- [ ] Implement a compact log-linear weighted trend estimator with a residual interval and a calendar-profile monthly allocator.
- [ ] Re-run targeted tests and verify they pass.

### Task 2: Implement regional daily model comparison and scenario forecasting

**Files:**
- Modify: `src/models/question2_forecasting.py`
- Modify: `tests/test_question2_forecasting.py`

- [ ] Write failing tests for deterministic calendar features, direct multi-step recursive forecasts, quantile ordering, and a weather-shock scenario that only changes the weather-sensitive forecast component.
- [ ] Run the targeted tests and verify they fail for the missing functions.
- [ ] Implement two candidates: seasonal-calendar ridge baseline and dynamic ridge with lagged pressure/search plus calendar/weather terms; select by sampled-observation rolling MAE and report RMSE, sMAPE, PICP and interval width.
- [ ] Re-run targeted tests and verify they pass.

### Task 3: Build the reproducible Question 2 artifact generator

**Files:**
- Create: `scripts/build_q2_forecasts.py`
- Create: `src/analysis/q2_forecast_outputs.py`
- Modify: `tests/test_question2_forecasting.py`

- [ ] Write failing tests for the output builder: required files, annual-to-monthly total constraint, three regions, forecast-only 2026 dates, and explicit estimate/provenance labels.
- [ ] Run the targeted tests and verify they fail.
- [ ] Implement input preparation, 2025 holdout evaluation, 2026 outputs, July-August peak range, and baseline/rain/heat scenario tables.
- [ ] Re-run targeted tests and verify they pass.

### Task 4: Add paper-ready diagnostics and verify the full project

**Files:**
- Modify: `src/analysis/q2_forecast_outputs.py`
- Modify: `tests/test_question2_forecasting.py`

- [ ] Write failing tests that verify diagnostic charts are generated from output tables and do not label estimated regional values as observed counts.
- [ ] Run the targeted tests and verify they fail.
- [ ] Generate a compact model-comparison table and two figures: city monthly forecast plus interval, and regional summer daily forecast plus scenario band.
- [ ] Run the Question 2 tests, then the full test suite; inspect produced table invariants and report paths.
