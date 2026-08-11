"""Attach transparent VOT-valued traveller time-loss diagnostics to plans."""

from __future__ import annotations

import pandas as pd

from src.models.value_of_time import estimate_traveler_time_loss


_REQUIRED_PLAN_COLUMNS = {
    "visitor_estimate_q50", "parking_unserved_spaces", "average_party_size",
    "transfer_share", "shuttle_shortfall", "recommended_shuttle_vehicles",
    "staff_shortfall", "recommended_total_staff_shifts",
}
_REQUIRED_VOT_COLUMNS = {
    "vot_scenario", "vot_cny_per_hour", "parking_delay_hours",
    "shuttle_delay_hours", "staff_delay_hours",
}


def attach_vot_costs(plan: pd.DataFrame, vot_scenarios: pd.DataFrame) -> pd.DataFrame:
    """Return plan with low/central/high visitor-time-loss estimates in CNY.

    The resource-selection objective is deliberately not changed: resource unit
    costs are not available as a verified local cash ledger.  These columns are
    traveller time-loss diagnostics under transparent scenario assumptions.
    """
    if missing := _REQUIRED_PLAN_COLUMNS - set(plan.columns):
        raise ValueError(f"plan missing columns: {sorted(missing)}")
    if missing := _REQUIRED_VOT_COLUMNS - set(vot_scenarios.columns):
        raise ValueError(f"vot_scenarios missing columns: {sorted(missing)}")
    if vot_scenarios["vot_scenario"].duplicated().any():
        raise ValueError("vot_scenarios must contain one row per vot_scenario")

    result = plan.copy()
    normalized = vot_scenarios.set_index("vot_scenario")
    if "central" not in normalized.index:
        raise ValueError("vot_scenarios must include central")
    for scenario, vot in normalized.iterrows():
        records = result.apply(
            lambda row: estimate_traveler_time_loss(
                daily_visitors=float(row["visitor_estimate_q50"]),
                parking_unserved_spaces=float(row["parking_unserved_spaces"]),
                average_party_size=float(row["average_party_size"]),
                parking_delay_hours=float(vot["parking_delay_hours"]),
                transfer_share=float(row["transfer_share"]),
                shuttle_shortfall_fraction=float(row["shuttle_shortfall"]) / max(float(row["recommended_shuttle_vehicles"]), 1.0),
                shuttle_delay_hours=float(vot["shuttle_delay_hours"]),
                staff_shortfall_fraction=float(row["staff_shortfall"]) / max(float(row["recommended_total_staff_shifts"]), 1.0),
                staff_delay_hours=float(vot["staff_delay_hours"]),
                vot_cny_per_hour=float(vot["vot_cny_per_hour"]),
            ),
            axis=1,
        ).apply(pd.Series)
        if scenario == "central":
            result["traveler_time_loss_hours"] = records["traveler_time_loss_hours"]
        result[f"traveler_time_loss_cny_{scenario}"] = records["traveler_time_loss_cny"]
    result["vot_cost_status"] = "traveler_time_loss_scenario_estimate_not_operator_cash_ledger"
    return result
