import pandas as pd
import subprocess
import sys
from pathlib import Path

from src.models.pressure_baseline import assign_calendar_regime, fit_pressure_baseline, rolling_time_validation


def test_assign_calendar_regime_uses_summer_and_statutory_holidays():
    calendar = pd.DataFrame(
        {
            "date": ["2024-01-10", "2024-05-01", "2024-07-10"],
            "是否法定放假": [0, 1, 0],
        }
    )

    result = assign_calendar_regime(calendar)

    assert result["regime"].tolist() == ["normal", "high_pressure", "high_pressure"]


def test_pressure_baseline_outputs_relative_index_and_uncertainty_not_visitor_counts():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02", "2024-07-01", "2024-07-02"],
            "region_code": ["BDH", "BDH", "SHG", "SHG"],
            "is_observed": [True, True, True, True],
            "visitor_index": [4.0, 6.0, 3.0, 5.0],
        }
    )
    calendar = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "是否法定放假": [0, 0],
            "是否周末": [0, 0],
        }
    )
    weather = pd.DataFrame(
        {"date": ["2024-07-01", "2024-07-02"], "日均气温_℃": [25.0, 27.0], "日降水量_mm": [0.0, 3.0]}
    )
    search = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "theme_attraction_lag_1": [0.1, 0.2],
            "theme_destination_lag_1": [0.3, 0.4],
        }
    )

    result = fit_pressure_baseline(panel, calendar, weather, search, ridge_alpha=1.0)

    assert len(result) == 4
    assert {"pressure_index", "uncertainty_proxy", "regime", "estimate_label"}.issubset(result.columns)
    assert (result["pressure_index"] > 0).all()
    assert result["estimate_label"].eq("relative_pressure_index").all()
    assert "visitor_estimate" not in result.columns


def test_pressure_baseline_can_extend_to_requested_unobserved_days():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "region_code": ["BDH", "BDH"],
            "is_observed": [True, True],
            "visitor_index": [4.0, 6.0],
        }
    )
    calendar = pd.DataFrame(
        {"date": ["2024-07-01", "2024-07-02", "2024-07-03"], "是否法定放假": [0, 0, 0]}
    )
    prediction_frame = pd.DataFrame(
        {"date": ["2024-07-01", "2024-07-02", "2024-07-03"], "region_code": ["BDH", "BDH", "BDH"]}
    )

    result = fit_pressure_baseline(
        panel,
        calendar,
        pd.DataFrame({"date": calendar["date"]}),
        pd.DataFrame({"date": calendar["date"]}),
        prediction_frame=prediction_frame,
    )

    assert len(result) == 3
    assert result["date"].max() == pd.Timestamp("2024-07-03")
    assert result.loc[result["date"] == pd.Timestamp("2024-07-03"), "observed_visitor_index"].isna().all()


def test_rolling_time_validation_trains_before_each_cutoff_only():
    panel = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02", "2024-07-03", "2024-07-04"],
            "region_code": ["BDH", "BDH", "BDH", "BDH"],
            "is_observed": [True, True, True, True],
            "visitor_index": [2.0, 3.0, 5.0, 7.0],
        }
    )
    calendar = pd.DataFrame({"date": panel["date"], "是否法定放假": [0, 0, 0, 0]})

    scores = rolling_time_validation(
        panel,
        calendar,
        pd.DataFrame({"date": panel["date"]}),
        pd.DataFrame({"date": panel["date"]}),
        cutoffs=["2024-07-02"],
    )

    assert len(scores) == 1
    assert scores.loc[0, "train_end"] == pd.Timestamp("2024-07-02")
    assert scores.loc[0, "test_rows"] == 2
    assert scores.loc[0, "mae"] >= 0


def test_pressure_build_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_pressure_baseline.py"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
