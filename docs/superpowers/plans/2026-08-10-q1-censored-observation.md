# Q1 Censored Observation Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible left-censored ranking-observation diagnostic layer to Question 1 without creating unobserved attraction visitor values.

**Architecture:** A focused diagnostics module will aggregate the existing sampled-date attraction panel by `date × region_code`. It will evaluate observed log-index density and left-censored probability under a normal approximation, while preserving the existing continuous pressure baseline and anchor-constrained visitor-scale outputs unchanged. A build script will write the diagnostic CSV and quality report alongside existing model inputs.

**Tech Stack:** Python, pandas, numpy, existing direct-function test convention, JSON quality reports.

---

### Task 1: Define and test censored observation diagnostics

**Files:**
- Create: `src/features/censored_diagnostics.py`
- Create: `tests/test_censored_diagnostics.py`

- [ ] **Step 1: Write failing tests for a mixed observed/censored sampled group and an invalid censored-only group**

```python
def test_diagnostics_use_minimum_observed_index_as_censor_threshold():
    panel = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-01"],
        "region_code": ["BDH", "BDH", "BDH"],
        "is_observed": [True, True, False],
        "visitor_index": [3.0, 7.0, None],
    })
    result = build_censored_observation_diagnostics(panel)
    row = result.iloc[0]
    assert row["density_count"] == 2
    assert row["left_censored_count"] == 1
    assert row["censor_threshold_index"] == 3.0
    assert row["uncertainty_flag"] == "small_observed_sample"

def test_diagnostics_do_not_estimate_threshold_for_censored_only_group():
    panel = pd.DataFrame({
        "date": ["2024-07-01"],
        "region_code": ["BDH"],
        "is_observed": [False],
        "visitor_index": [None],
    })
    result = build_censored_observation_diagnostics(panel)
    assert result.loc[0, "uncertainty_flag"] == "no_density_observation"
    assert pd.isna(result.loc[0, "total_log_likelihood"])
```

- [ ] **Step 2: Run the direct test harness and confirm import failure**

Run:

```powershell
$py='C:\Users\lyy20\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -c "from tests.test_censored_diagnostics import test_diagnostics_use_minimum_observed_index_as_censor_threshold; test_diagnostics_use_minimum_observed_index_as_censor_threshold()"
```

Expected: `ModuleNotFoundError` for `src.features.censored_diagnostics`.

- [ ] **Step 3: Implement the minimal diagnostics function**

Implement `build_censored_observation_diagnostics(panel, min_log_std=0.25)` with required input columns `date`, `region_code`, `is_observed`, and `visitor_index`.

- Convert dates and visitor indices safely.
- Group only by actual panel rows; do not generate dates.
- For valid groups, set the threshold to the minimum positive observed `visitor_index`.
- Compute `log_index_mean`, sample standard deviation with lower bound `min_log_std`, normal-density contributions for observed rows, normal-CDF log-probability times censored count, and their sum.
- For zero observed rows, emit counts but leave threshold and likelihood fields null with `uncertainty_flag=no_density_observation`.
- For one or two observed rows, use the standard deviation lower bound and `uncertainty_flag=small_observed_sample`; otherwise use `estimated_log_std`.

- [ ] **Step 4: Re-run both tests and verify they pass**

Run:

```powershell
& $py -c "from tests.test_censored_diagnostics import test_diagnostics_use_minimum_observed_index_as_censor_threshold, test_diagnostics_do_not_estimate_threshold_for_censored_only_group; test_diagnostics_use_minimum_observed_index_as_censor_threshold(); test_diagnostics_do_not_estimate_threshold_for_censored_only_group(); print('censored diagnostics tests passed')"
```

Expected: `censored diagnostics tests passed`.

### Task 2: Build reproducible Q1 diagnostic outputs

**Files:**
- Create: `scripts/build_censored_observation_diagnostics.py`
- Modify: `data/model_input/README.md`
- Modify: `tests/test_censored_diagnostics.py`

- [ ] **Step 1: Add a failing build-script test**

```python
def test_censored_diagnostics_build_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_censored_observation_diagnostics.py"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run the build-script test and confirm it fails because the script is absent**

Run:

```powershell
& $py -c "from tests.test_censored_diagnostics import test_censored_diagnostics_build_script_runs_from_project_root; test_censored_diagnostics_build_script_runs_from_project_root()"
```

Expected: missing-file assertion.

- [ ] **Step 3: Implement the build script and documentation**

The script must:

- insert the project root into `sys.path` before importing `src`;
- read `data/model_input/censored_attraction_observation_panel.csv`;
- write `data/model_input/censored_observation_diagnostics.csv` in UTF-8-SIG;
- write `data/model_input/censored_observation_quality_report.json` with group count, valid-likelihood group count, no-density group count, small-sample group count, and censor-rate summary;
- print the JSON report; and
- document both files, their non-imputation interpretation, and rebuild command in `data/model_input/README.md`.

- [ ] **Step 4: Run the build-script test and inspect output columns**

Run:

```powershell
& $py scripts\build_censored_observation_diagnostics.py
Import-Csv data\model_input\censored_observation_diagnostics.csv | Select-Object -First 5
```

Expected: output includes `density_count`, `left_censored_count`, `censor_threshold_index`, `total_log_likelihood`, and `uncertainty_flag`.

### Task 3: Verify no regression in the Question 1 foundation

**Files:**
- Modify: none

- [ ] **Step 1: Run all existing direct tests plus new diagnostics tests**

Run the project’s direct test harness to cover search features, anchor preparation, observation panel construction, pressure baseline, scale calibration, and censored diagnostics.

Expected: every function returns successfully and prints `all q1 foundation tests passed`.

- [ ] **Step 2: Check output boundaries manually**

- Verify the diagnostics date range is a subset of `censored_attraction_observation_panel.csv` dates.
- Verify every valid group satisfies `density_count + left_censored_count =` the panel group size.
- Verify no output contains a per-attraction estimate for a `left_censored` input row.
- Verify existing `daily_region_visitor_scale_estimates_2023_2025.csv` remains labelled `anchor_constrained_estimate`.

- [ ] **Step 3: Report the diagnostic interpretation**

Summarize the number of valid groups, small-sample groups, censored-only groups, and the rule that this diagnostic layer corrects selection bias only within actual ranking sample dates.
