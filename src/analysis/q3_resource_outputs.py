"""Translate Question 2 regional forecasts into auditable Question 3 tier recommendations."""

from __future__ import annotations

import pandas as pd

from src.models.question3_resource_optimization import allocate_period_demand, select_resource_tier


def build_regime_recommendations(forecast: pd.DataFrame, plans: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """Choose a tier for each region and season using conservative (q90) demand."""
    frame = forecast.copy(); frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["regime"] = pd.cut(frame["date"].dt.month, bins=[0, 2, 6, 12], labels=["off_season", "shoulder_season", "peak_season"])
    rows = []
    for (region, regime), group in frame.groupby(["region_code", "regime"], observed=True):
        demand = allocate_period_demand(float(group["visitor_estimate_q90"].max()), [0.25, 0.50, 0.25], 0.35, 3.0, 0.15)
        tier, diag = select_resource_tier(demand, plans[str(region)], risk_penalty=10.0)
        rows.append({"region_code": region, "regime": str(regime), "design_daily_visitors_q90": float(group["visitor_estimate_q90"].max()), "recommended_tier": tier, "objective": float(diag.iloc[0]["objective"]), "bottleneck_risk": str(diag.iloc[0][["parking_risk", "shuttle_risk", "entry_risk"]].astype(float).idxmax())})
    return pd.DataFrame(rows)
