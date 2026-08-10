import pandas as pd

from src.features.anchor_selection import classify_direct_region_anchors


def test_anchor_selection_keeps_only_direct_region_point_anchors():
    anchors = pd.DataFrame(
        {
            "region_code": ["BDH", "BDH", "CITY", "SHG"],
            "period_start": ["2024-01-01"] * 4,
            "period_end": ["2024-01-02"] * 4,
            "anchor_use": ["point_calibration", "point_calibration", "point_calibration", "lower_bound_only"],
            "metric_scope": [
                "district_tourist_visitors",
                "paid_scenic_gate_entries",
                "citywide_tourist_visitors",
                "district_tourist_visitors",
            ],
            "facility_or_scope": ["all_visitors", "Pigeon_Nest_Park", "all_visitors", "all_visitors"],
        }
    )

    result = classify_direct_region_anchors(
        anchors,
        target_regions={"BDH", "HGA", "SHG"},
        model_start="2023-01-01",
        model_end="2025-12-31",
    )

    assert result["scope_decision"].tolist() == [
        "direct_region_scale",
        "subarea_or_scenic_scope",
        "outside_target_regions",
        "lower_bound_only",
    ]
