# Q1 Hierarchical Censored Likelihood Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Question 1 ridge-only pressure estimator with a dynamic, hierarchical, left-censored likelihood model and package reproducible Q1 outputs.

**Architecture:** A new model module will estimate sampled-day regional latent pressure, attraction effects, regional variance and covariate coefficients jointly by penalized negative log likelihood. It will use density terms for observed ranked attractions and normal-CDF censor terms for same-day unranked attractions. A separate build script will generate continuous 2023–2025 pressures, model diagnostics and a comparison with the existing ridge baseline before reusing the current audited anchor-scale layer.

**Tech Stack:** Python, pandas, numpy, SciPy optimizer, existing direct-function tests, JSON/CSV reports.

---

### Task 1: Build the censored-likelihood objective

**Files:**
- Create: `src/models/hierarchical_censored_pressure.py`
- Create: `tests/test_hierarchical_censored_pressure.py`

- [ ] Write failing tests for: (a) observed density plus censored rows yields finite loss, (b) removing a censored row changes loss, and (c) a censored-only group is excluded from the likelihood.
- [ ] Run direct tests and confirm `ModuleNotFoundError`.
- [ ] Implement `prepare_censored_likelihood_data(panel, covariates)` returning numeric arrays for observed `z`, censor thresholds, sampled-day group ids, attraction ids, region ids and feature matrix.
- [ ] Implement `negative_log_posterior(params, prepared, lambda_alpha, lambda_dynamic, lambda_beta)` using:

```python
loss = -density_log_likelihood - censor_log_likelihood
loss += lambda_alpha * np.sum(alpha ** 2)
loss += lambda_dynamic * np.sum((u_current - rho * u_previous) ** 2)
loss += lambda_beta * np.sum(beta ** 2)
```

- [ ] Enforce per-region centered attraction effects after unpacking parameters and clip normal-CDF probabilities below `1e-12` before `log`.
- [ ] Re-run objective tests and require finite numeric results.

### Task 2: Fit sampled-day hierarchical model

**Files:**
- Modify: `src/models/hierarchical_censored_pressure.py`
- Modify: `tests/test_hierarchical_censored_pressure.py`

- [ ] Write a failing synthetic fit test requiring `fit_hierarchical_censored_pressure(...)` to return `converged`, sampled `censored_likelihood_pressure_index`, attraction effects and final objective.
- [ ] Run it and confirm the function is absent.
- [ ] Implement L-BFGS-B fitting with parameters: covariate coefficients, two region dummy effects, sampled-day residuals, attraction effects, log regional standard deviations and transformed AR coefficient `tanh(raw_rho)`.
- [ ] Use one deterministic initialization from the current ridge baseline; reject a result unless the optimizer succeeds and all fitted standard deviations are positive.
- [ ] Return sampled-day pressure records with `estimate_label='censored_likelihood_pressure_index'`, plus parameter and convergence diagnostics.
- [ ] Re-run synthetic fit tests and assert censored likelihood is finite and the output has no absolute-visitor field.

### Task 3: Generate continuous-day Q1 pressure and comparison

**Files:**
- Create: `scripts/build_hierarchical_censored_pressure.py`
- Modify: `tests/test_hierarchical_censored_pressure.py`
- Modify: `data/model_input/README.md`

- [ ] Write a failing build-script test invoking `scripts/build_hierarchical_censored_pressure.py` from project root.
- [ ] Implement the script to read the censored panel, calendar, weather and lagged search features; fit on actual sampled dates; and generate all dates in 2023–2025 for BDH/HGA/SHG.
- [ ] Write:
  - `daily_region_censored_likelihood_pressure_2023_2025.csv`
  - `censored_likelihood_parameter_diagnostics.csv`
  - `censored_likelihood_quality_report.json`
  - `censored_likelihood_vs_ridge_validation.csv`
- [ ] Generate rolling comparisons at `2023-06-30` and `2024-06-30`, reporting each model's MAE, valid censored-group count and final likelihood.
- [ ] Re-run build-script test; inspect that the continuous output has `3 × 1096 = 3288` rows and only the relative-pressure label.

### Task 4: Recalibrate daily visitor scenarios from the main pressure model

**Files:**
- Create: `scripts/build_censored_likelihood_visitor_scale.py`
- Modify: `tests/test_hierarchical_censored_pressure.py`
- Modify: `data/model_input/README.md`

- [ ] Write a failing script test for the new scale build.
- [ ] Implement the script to read `daily_region_censored_likelihood_pressure_2023_2025.csv` and `calibration_anchor_scope_ledger.csv`, call the existing `calibrate_daily_visitor_scale`, and write:
  - `daily_region_visitor_scale_censored_likelihood_2023_2025.csv`
  - `visitor_scale_censored_likelihood_anchor_fit.csv`
  - `visitor_scale_censored_likelihood_quality_report.json`
- [ ] Preserve `primary_scale` versus `diagnostic_only`, carry-forward labels and conservative/baseline/high scenarios.
- [ ] Re-run script test and verify no CITY, ARA or BND row enters the output regions.

### Task 5: Package Question 1 for handoff

**Files:**
- Create: `问题一_模型说明与封装.md`
- Modify: `data/model_input/CSV文件作用说明.md`

- [ ] Write `问题一_模型说明与封装.md` covering input provenance, model equations, censoring rule, identifiability constraints, optimizer convergence, validation comparison, anchor conflict handling, uncertainty interpretation and exact rebuild order.
- [ ] Update the CSV guide to identify the censored-likelihood outputs as the Question 1 main chain and ridge outputs as the comparison baseline.
- [ ] Run all existing Q1 tests plus the new hierarchical tests. Require a single successful completion message.
- [ ] Manually verify: sampled-only censor diagnostics, 3,288 continuous pressure rows, no absolute count claim before anchor scaling, and explicit labels on all external/carry-forward uncertainty.
