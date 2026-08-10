import pandas as pd

from src.features.anchor_calibration import calibrate_absolute_scale


def test_anchor_calibration_respects_coverage_share_not_full_city_total() -> None:
    pressure = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-07-01", "2024-07-02"]),
            "region_code": ["BDH", "BDH"],
            "pressure_index": [1.0, 2.0],
        }
    )
    anchors = pd.DataFrame(
        {"region_code": ["BDH"], "period_start": ["2024-07-01"], "period_end": ["2024-07-02"], "visitor_total": [300]}
    )

    result = calibrate_absolute_scale(pressure, anchors, city_annual_total=1000, coverage_share=0.5)

    assert result["estimate_label"].eq("anchor_constrained_estimate").all()
    assert result["visitor_estimate_baseline"].sum() == 300
    assert result["city_coverage_share"].eq(0.5).all()
