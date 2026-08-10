import pandas as pd

from src.features.anchor_preparation import prepare_calibration_anchors


def test_prepare_calibration_anchors_converts_daily_average_and_flags_lower_bound() -> None:
    frame = pd.DataFrame(
        {
            "period_start": ["2024-07-01", "2024-07-01"],
            "period_end": ["2024-07-02", "2024-07-14"],
            "region_name": ["Beidaihe_District", "Beidaihe_District"],
            "visitor_count_10k_persons": [5.0, 58.0],
            "frequency": ["daily_average", "cumulative_to_date"],
            "value_qualifier": ["approximately", "greater_than"],
            "data_status": ["verified", "verified"],
        }
    )

    result = prepare_calibration_anchors(frame, {"Beidaihe_District": "BDH"})

    assert result.loc[0, "visitor_total"] == 100_000
    assert result.loc[0, "anchor_use"] == "point_calibration"
    assert result.loc[1, "anchor_use"] == "lower_bound_only"


def test_prepare_calibration_anchors_recognizes_lower_bound_in_frequency_label() -> None:
    frame = pd.DataFrame(
        {
            "period_start": ["2024-01-01"],
            "period_end": ["2024-07-31"],
            "region_name": ["Beidaihe_District"],
            "visitor_count_10k_persons": [900.0],
            "frequency": ["year_to_date_lower_bound"],
            "data_status": ["verified"],
        }
    )

    result = prepare_calibration_anchors(frame, {"Beidaihe_District": "BDH"})

    assert result.loc[0, "anchor_use"] == "lower_bound_only"
