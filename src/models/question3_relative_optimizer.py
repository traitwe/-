"""Multi-objective optimization over relative service tiers for Question 3."""

from __future__ import annotations

from itertools import product

import pandas as pd


RESOURCE_NAMES = ("staff", "entry", "parking_guidance", "shuttle")
DEFAULT_COST = {"staff": 1.0, "entry": 0.8, "parking_guidance": 0.6, "shuttle": 1.2}
DEFAULT_RISK_WEIGHT = {"staff": 1.0, "entry": 1.0, "parking_guidance": 0.8, "shuttle": 1.2}


def _required_tier(relative_pressure: float, resource: str) -> int:
    thresholds = {"staff": (1.0, 1.4), "entry": (0.9, 1.6), "parking_guidance": (0.8, 1.2), "shuttle": (1.1, 1.8)}
    low, high = thresholds[resource]
    if relative_pressure < low:
        return 0
    if relative_pressure < high:
        return 1
    return 2


def optimize_relative_tiers(relative_pressure: float, risk_penalty: float = 5.0, budget: float | None = None, anchor_support: dict[str, float] | None = None) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Enumerate 0/1/2 resource tiers and minimize relative cost plus standardized risk."""
    if relative_pressure < 0 or risk_penalty < 0:
        raise ValueError("relative_pressure and risk_penalty must be non-negative")
    if budget is not None and budget < 0:
        raise ValueError("budget must be non-negative")
    support = {resource: 0.0 for resource in RESOURCE_NAMES}
    for resource, value in (anchor_support or {}).items():
        if resource not in support or not 0 <= value <= 1:
            raise ValueError("anchor support must target a known resource and stay in [0, 1]")
        support[resource] = value
    targets = {resource: _required_tier(relative_pressure, resource) for resource in RESOURCE_NAMES}
    records = []
    for levels in product(range(3), repeat=len(RESOURCE_NAMES)):
        row = {f"{resource}_tier": level for resource, level in zip(RESOURCE_NAMES, levels)}
        cost = sum(DEFAULT_COST[resource] * level for resource, level in zip(RESOURCE_NAMES, levels))
        risks = {f"{resource}_risk": DEFAULT_RISK_WEIGHT[resource] * max(targets[resource] - level, 0) / 2.0 * (1 - support[resource]) for resource, level in zip(RESOURCE_NAMES, levels)}
        risk = sum(risks.values())
        row.update(risks); row.update({"target_tier": max(targets.values()), "relative_cost": cost, "standardized_service_risk": risk, "objective": cost + risk_penalty * risk})
        records.append(row)
    diagnostics = pd.DataFrame(records)
    if budget is not None:
        diagnostics = diagnostics.loc[diagnostics["relative_cost"] <= budget].copy()
    diagnostics = diagnostics.sort_values(["objective", "relative_cost"]).reset_index(drop=True)
    return diagnostics.iloc[0].to_dict(), diagnostics
