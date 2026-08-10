import pandas as pd

from src.analysis.q3_resource_outputs import build_regime_recommendations


def test_regime_recommendations_cover_each_region_and_three_seasons():
    forecast = pd.DataFrame({"date": ["2026-01-01", "2026-05-01", "2026-07-01"] * 3, "region_code": ["BDH"] * 3 + ["HGA"] * 3 + ["SHG"] * 3, "visitor_estimate_q90": [100, 200, 300] * 3})
    plan = {"parking_spaces": 100, "parking_turnover": 1, "shuttle_headway_minutes": 15, "shuttle_seats": 40, "shuttle_load_factor": 0.8, "period_hours": 4, "entry_capacity": 200, "tier_cost": 1}
    plans = {region: {"normal": plan, "surge": {**plan, "parking_spaces": 200, "entry_capacity": 400, "tier_cost": 2}} for region in ["BDH", "HGA", "SHG"]}

    result = build_regime_recommendations(forecast, plans)

    assert len(result) == 9
    assert set(result["recommended_tier"]).issubset({"normal", "surge"})
