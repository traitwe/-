"""Unit-consistent resource allocation primitives for Question 3."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def allocate_period_demand(daily_visitors: float, period_shares: Iterable[float], car_share: float, persons_per_car: float, shuttle_share: float) -> pd.DataFrame:
    """Split visitor demand by service period and convert only within matching units."""
    shares = np.asarray(list(period_shares), dtype=float)
    if daily_visitors < 0 or len(shares) == 0 or (shares < 0).any() or shares.sum() <= 0:
        raise ValueError("daily visitors and period shares must be non-negative with positive total share")
    if not 0 <= car_share <= 1 or not 0 <= shuttle_share <= 1 or persons_per_car <= 0:
        raise ValueError("invalid travel-mode parameters")
    visitors = float(daily_visitors) * shares / shares.sum()
    return pd.DataFrame({
        "period": ["morning", "midday", "evening"][:len(shares)],
        "visitor_demand": visitors,
        "parking_vehicle_demand": visitors * car_share / persons_per_car,
        "shuttle_passenger_demand": visitors * shuttle_share,
    })


def evaluate_resource_plan(demand: pd.DataFrame, plan: Mapping[str, float]) -> pd.DataFrame:
    """Evaluate parking, shuttle and entry plans without aggregating incompatible units."""
    required = {"period", "visitor_demand", "parking_vehicle_demand", "shuttle_passenger_demand"}
    if required.difference(demand.columns):
        raise ValueError("demand table is incomplete")
    for key in ["parking_spaces", "parking_turnover", "shuttle_headway_minutes", "shuttle_seats", "shuttle_load_factor", "period_hours", "entry_capacity"]:
        if key not in plan or float(plan[key]) <= 0:
            raise ValueError(f"plan parameter must be positive: {key}")
    result = demand.copy()
    result["parking_capacity_vehicles"] = float(plan["parking_spaces"]) * float(plan["parking_turnover"])
    result["shuttle_capacity_passengers"] = float(plan["period_hours"]) * 60.0 / float(plan["shuttle_headway_minutes"]) * float(plan["shuttle_seats"]) * float(plan["shuttle_load_factor"])
    result["entry_capacity_visitors"] = float(plan["entry_capacity"])
    result["parking_shortfall_vehicles"] = (result["parking_vehicle_demand"] - result["parking_capacity_vehicles"]).clip(lower=0)
    result["shuttle_shortfall_passengers"] = (result["shuttle_passenger_demand"] - result["shuttle_capacity_passengers"]).clip(lower=0)
    result["entry_shortfall_visitors"] = (result["visitor_demand"] - result["entry_capacity_visitors"]).clip(lower=0)
    return result


def select_resource_tier(demand: pd.DataFrame, plans: Mapping[str, Mapping[str, float]], risk_penalty: float = 1.0) -> tuple[str, pd.DataFrame]:
    """Select a resource tier using declared cost plus unit-normalized service-risk penalties."""
    if risk_penalty < 0 or not plans:
        raise ValueError("risk_penalty must be non-negative and plans must not be empty")
    records: list[dict[str, float | str]] = []
    for tier, plan in plans.items():
        if "tier_cost" not in plan:
            raise ValueError("every plan requires tier_cost")
        evaluated = evaluate_resource_plan(demand, plan)
        parking_risk = float(evaluated["parking_shortfall_vehicles"].sum() / max(evaluated["parking_vehicle_demand"].sum(), 1e-9))
        shuttle_risk = float(evaluated["shuttle_shortfall_passengers"].sum() / max(evaluated["shuttle_passenger_demand"].sum(), 1e-9))
        entry_risk = float(evaluated["entry_shortfall_visitors"].sum() / max(evaluated["visitor_demand"].sum(), 1e-9))
        service_risk = parking_risk + shuttle_risk + entry_risk
        records.append({"tier": str(tier), "tier_cost": float(plan["tier_cost"]), "parking_risk": parking_risk, "shuttle_risk": shuttle_risk, "entry_risk": entry_risk, "service_risk": service_risk, "objective": float(plan["tier_cost"]) + risk_penalty * service_risk})
    diagnostics = pd.DataFrame(records).sort_values(["objective", "tier_cost", "tier"]).reset_index(drop=True)
    return str(diagnostics.loc[0, "tier"]), diagnostics
