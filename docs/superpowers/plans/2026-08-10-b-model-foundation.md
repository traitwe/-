# B Model Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build reproducible model-ready features and calibration inputs for the Qinhuangdao B problem without treating proxies as observed visitor counts.

**Architecture:** Create focused feature builders for search heat and ranked-attraction observations, then create an anchor-calibration module that emits only estimated absolute-scale scenarios. Keep all source data immutable and persist quality metadata beside each derived table.

**Tech Stack:** Python, pandas, numpy, existing `src/data/cleaning.py` conventions, direct test functions in `tests/test_cleaning.py`.

---

### Task 1: Build annually standardized search-theme features

**Files:**
- Create: `src/features/search_features.py`
- Modify: `tests/test_search_features.py`

- [ ] Write failing tests for year-wise z-scoring, weather-keyword exclusion, and A/B/C/D keyword filtering.
- [ ] Run the direct test harness and confirm import failure.
- [ ] Implement `build_search_theme_features(frame, keyword_rules)` returning yearly standardized, lagged, 2--4 theme factors plus quality flags.
- [ ] Re-run tests and verify each passes.

### Task 2: Encode A3b ranking truncation

**Files:**
- Create: `src/features/ranked_observation.py`
- Modify: `tests/test_ranked_observation.py`

- [ ] Write failing tests showing observed rows become density observations and unobserved attraction-date pairs become censored observations.
- [ ] Confirm tests fail because the module is absent.
- [ ] Implement `build_censored_observation_panel` with an `is_observed` flag and month-level threshold group; do not impute visitor indices.
- [ ] Re-run tests and verify each passes.

### Task 3: Build anchor calibration scenarios

**Files:**
- Create: `src/features/anchor_calibration.py`
- Modify: `tests/test_anchor_calibration.py`

- [ ] Write failing tests for annual-total reconciliation, regional-anchor calibration, and interval-labelled output.
- [ ] Confirm tests fail because the module is absent.
- [ ] Implement `calibrate_absolute_scale` to return conservative/baseline/high estimates labelled `anchor_constrained_estimate`.
- [ ] Re-run tests and verify each passes.

### Task 4: Produce model-ready inputs and profile

**Files:**
- Create: `scripts/build_b_model_inputs.py`
- Create: `data/model_input/README.md`
- Modify: `tests/test_model_input_build.py`

- [ ] Write a failing integration test that requires the three derived CSVs and a quality report.
- [ ] Implement the build script using only `data/clean` outputs.
- [ ] Run all feature tests and the build script; inspect row counts, date coverage, and prohibited-field flags.

### Task 5: Implement the first identifiable baseline

**Files:**
- Create: `src/models/pressure_baseline.py`
- Create: `tests/test_pressure_baseline.py`

- [ ] Write failing tests for two-state calendar regime assignment and pressure-index output without absolute-count claims.
- [ ] Implement a transparent regularized baseline before any Bayesian sampler.
- [ ] Verify rolling-time validation and labelled uncertainty proxy outputs.

