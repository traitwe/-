import pandas as pd

from src.models.question3_relative_tiers import classify_relative_pressure, beidaihe_summer_trigger


def test_relative_pressure_classifies_within_region_not_against_external_capacity():
    source = pd.DataFrame({"region_code": ["BDH"] * 4, "visitor_estimate_q50": [100.0, 120.0, 160.0, 220.0]})

    result = classify_relative_pressure(source)

    assert result["pressure_tier"].tolist() == ["low", "normal", "high", "extreme"]
    assert result["relative_pressure"].iloc[0] < 1


def test_beidaihe_summer_trigger_uses_percentile_and_calendar_conditions():
    assert beidaihe_summer_trigger(relative_pressure=1.5, is_weekend=False, is_holiday=False) == "surge"
    assert beidaihe_summer_trigger(relative_pressure=2.0, is_weekend=True, is_holiday=False) == "emergency"
