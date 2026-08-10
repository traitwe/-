import pandas as pd
import subprocess
import sys
from pathlib import Path

from src.models.scale_calibration import calibrate_daily_visitor_scale


def test_scale_calibration_matches_a_single_direct_anchor_and_labels_interval():
    pressure = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "region_code": ["BDH", "BDH"],
            "pressure_index": [10.0, 20.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "region_code": ["BDH"],
            "period_start": ["2024-07-01"],
            "period_end": ["2024-07-02"],
            "visitor_total": [300.0],
            "scope_decision": ["direct_region_scale"],
        }
    )

    estimates, anchor_fit = calibrate_daily_visitor_scale(pressure, anchors)

    assert estimates["visitor_estimate_baseline"].round(6).tolist() == [100.0, 200.0]
    assert estimates["estimate_label"].eq("anchor_constrained_estimate").all()
    assert estimates["visitor_estimate_conservative"].round(6).tolist() == [80.0, 160.0]
    assert estimates["visitor_estimate_high"].round(6).tolist() == [120.0, 240.0]
    assert round(anchor_fit.loc[0, "fitted_period_total"], 6) == 300.0


def test_scale_calibration_prioritizes_annual_anchor_over_conflicting_short_period_anchor():
    pressure = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-07-01"],
            "region_code": ["BDH", "BDH"],
            "pressure_index": [10.0, 20.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "region_code": ["BDH", "BDH"],
            "period_start": ["2024-01-01", "2024-07-01"],
            "period_end": ["2024-12-31", "2024-07-01"],
            "visitor_total": [300.0, 600.0],
            "scope_decision": ["direct_region_scale", "direct_region_scale"],
            "frequency": ["annual", "daily_peak"],
        }
    )

    estimates, anchor_fit = calibrate_daily_visitor_scale(pressure, anchors)

    assert round(estimates.loc[0, "visitor_estimate_baseline"], 6) == 100.0
    assert anchor_fit.loc[anchor_fit["frequency"].eq("annual"), "calibration_role"].iloc[0] == "primary_scale"
    assert anchor_fit.loc[anchor_fit["frequency"].eq("daily_peak"), "calibration_role"].iloc[0] == "diagnostic_only"


def test_scale_calibration_marks_years_without_anchor_as_carried_forward():
    pressure = pd.DataFrame(
        {
            "date": ["2024-07-01", "2025-07-01"],
            "region_code": ["BDH", "BDH"],
            "pressure_index": [10.0, 10.0],
        }
    )
    anchors = pd.DataFrame(
        {
            "region_code": ["BDH"],
            "period_start": ["2024-07-01"],
            "period_end": ["2024-07-01"],
            "visitor_total": [100.0],
            "scope_decision": ["direct_region_scale"],
        }
    )

    estimates, _ = calibrate_daily_visitor_scale(pressure, anchors)

    carried = estimates.loc[estimates["date"].eq(pd.Timestamp("2025-07-01"))].iloc[0]
    assert carried["scale_source_year"] == 2024
    assert carried["scale_interval_method"] == "carry_forward_scenario"


def test_scale_build_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_visitor_scale_estimates.py"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
