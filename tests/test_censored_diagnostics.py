import pandas as pd
import subprocess
import sys
from pathlib import Path

from src.features.censored_diagnostics import build_censored_observation_diagnostics


def test_diagnostics_use_minimum_observed_index_as_censor_threshold():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-01", "2024-07-01"],
            "region_code": ["BDH", "BDH", "BDH"],
            "is_observed": [True, True, False],
            "visitor_index": [3.0, 7.0, None],
        }
    )

    result = build_censored_observation_diagnostics(panel)
    row = result.iloc[0]

    assert row["density_count"] == 2
    assert row["left_censored_count"] == 1
    assert row["censor_threshold_index"] == 3.0
    assert row["uncertainty_flag"] == "small_observed_sample"
    assert pd.notna(row["total_log_likelihood"])


def test_diagnostics_do_not_estimate_threshold_for_censored_only_group():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01"],
            "region_code": ["BDH"],
            "is_observed": [False],
            "visitor_index": [None],
        }
    )

    result = build_censored_observation_diagnostics(panel)

    assert result.loc[0, "uncertainty_flag"] == "no_density_observation"
    assert pd.isna(result.loc[0, "total_log_likelihood"])


def test_diagnostics_keep_zero_index_rank_rows_in_observed_count_without_inventing_threshold():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-01"],
            "region_code": ["BDH", "BDH"],
            "is_observed": [True, False],
            "visitor_index": [0.0, None],
        }
    )

    result = build_censored_observation_diagnostics(panel)

    assert result.loc[0, "density_count"] == 1
    assert result.loc[0, "uncertainty_flag"] == "no_positive_density_index"
    assert pd.isna(result.loc[0, "censor_threshold_index"])


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
