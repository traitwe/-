import pandas as pd
import pytest

from src.models.question4_scenarios import apply_continuous_rain, apply_event_pulse, service_gap_index


def test_event_pulse_has_preheat_peak_and_tail_without_changing_other_dates():
    baseline = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=7).astype(str), "visitor_estimate": [100] * 7})
    result = apply_event_pulse(baseline, "visitor_estimate", "2026-07-04", 0.5)

    assert result.loc[3, "scenario_visitors"] == 150
    assert result.loc[2, "scenario_visitors"] == 125
    assert result.loc[0, "scenario_visitors"] == 100


def test_continuous_rain_reduces_each_successive_day_and_never_goes_negative():
    baseline = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=4).astype(str), "visitor_estimate": [100, 100, 100, 100]})
    result = apply_continuous_rain(baseline, "visitor_estimate", "2026-07-01", 4, 0.15)

    assert result["scenario_visitors"].tolist() == pytest.approx([85.0, 70.0, 55.0, 40.0])


def test_service_gap_is_positive_when_fixed_resources_are_short():
    gap = service_gap_index(
        parking_required=200, parking_available=100,
        shuttle_required=10, shuttle_available=5,
        staff_required=20, staff_available=10,
    )

    assert gap == 0.5
