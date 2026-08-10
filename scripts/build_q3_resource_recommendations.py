"""Build Question 3 seasonal resource-tier recommendations from Question 2 forecasts."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.q3_resource_outputs import build_regime_recommendations

forecast = pd.read_csv(ROOT / "outputs/question2_analysis/q2_region_daily_forecast_2026.csv", encoding="utf-8-sig")

def tier(parking_spaces, headway, entry, cost):
    return {"parking_spaces": parking_spaces, "parking_turnover": 1.0, "shuttle_headway_minutes": headway, "shuttle_seats": 40.0, "shuttle_load_factor": 0.75, "period_hours": 4.0, "entry_capacity": entry, "tier_cost": cost}

# HGA parking uses the 1,272-space Yutian anchor; SHG uses the 791-space access-buffer anchor.
# BDH capacity is a relative operating scenario because core-board totals are unpublished.
plans = {
    "BDH": {"normal": tier(350, 30, 6000, 1), "surge": tier(500, 15, 9000, 2), "emergency": tier(650, 10, 12000, 4)},
    "HGA": {"normal": tier(1272, 30, 5000, 1), "surge": tier(1272, 15, 7500, 2), "emergency": tier(1272, 10, 10000, 4)},
    "SHG": {"normal": tier(791, 30, 4500, 1), "surge": tier(791, 15, 7000, 2), "emergency": tier(1000, 10, 9000, 4)},
}
output = ROOT / "outputs/question3_analysis"; output.mkdir(parents=True, exist_ok=True)
result = build_regime_recommendations(forecast, plans)
result["capacity_parameter_status"] = "scenario_parameters__HGA_SHG_parking_anchored__BDH_core_parking_relative"
result.to_csv(output / "q3_seasonal_resource_recommendations.csv", index=False, encoding="utf-8-sig")
print(result.to_string(index=False))
