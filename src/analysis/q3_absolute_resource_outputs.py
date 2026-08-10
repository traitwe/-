"""Create transparent absolute resource planning values for Question 3."""

from __future__ import annotations

import math

import pandas as pd

from src.models.question3_absolute_capacity import minimum_staff, parking_plan, peak_parking_demand, required_shuttle_vehicles
from src.models.question3_staff_scheduling import optimize_three_shifts


BLOCK_SHARES = (0.25, 0.50, 0.25)
BLOCK_HOURS = 4.0


def _staff_schedule(daily_visitors: float, block_shares: tuple[float, float, float], service_rate: float, utilisation: float, minimum_people: int, multiplier: float = 1.0) -> dict[str, int]:
    demand = [
        minimum_staff(daily_visitors * share * multiplier / BLOCK_HOURS, service_rate, utilisation, minimum_people)
        for share in block_shares
    ]
    return optimize_three_shifts(demand)


def _shuttle_schedule(daily_visitors: float, row: pd.Series) -> tuple[int, dict[str, int]]:
    fleets = [
        required_shuttle_vehicles(
            daily_visitors * share * float(row["transfer_share"]) / BLOCK_HOURS,
            int(row["seats_per_vehicle"]),
            float(row["load_factor"]),
            float(row["round_trip_minutes"]),
            float(row["maximum_headway_minutes"]),
        )
        for share in BLOCK_SHARES
    ]
    return max(fleets), optimize_three_shifts(fleets)


def build_daily_absolute_plan(forecast: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Build daily planning recommendations from q50 visitors and scenario inputs.

    Inputs are deliberately explicit because car mode share, vehicle type, service
    productivity and unobserved capacity are planning assumptions rather than
    observed operational records.
    """
    required_forecast = {"date", "region_code", "visitor_estimate_q50"}
    required_scenario = {"region_code", "demand_scenario", "car_share", "average_party_size", "peak_arrival_share", "dwell_hours", "peak_window_hours", "transfer_share", "seats_per_vehicle", "load_factor", "round_trip_minutes", "maximum_headway_minutes", "entry_service_rate", "parking_service_rate", "utilisation_target", "permanent_spaces", "temporary_spaces"}
    if missing := required_forecast - set(forecast.columns):
        raise ValueError(f"forecast missing columns: {sorted(missing)}")
    if missing := required_scenario - set(scenarios.columns):
        raise ValueError(f"scenarios missing columns: {sorted(missing)}")
    merged = forecast.loc[:, ["date", "region_code", "visitor_estimate_q50"]].merge(scenarios, on="region_code", how="inner", validate="many_to_one")
    if len(merged) != len(forecast):
        raise ValueError("each forecast region requires exactly one scenario row")
    records: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        visitors = float(row["visitor_estimate_q50"])
        peak_spaces = peak_parking_demand(visitors, float(row["car_share"]), float(row["average_party_size"]), float(row["peak_arrival_share"]), float(row["dwell_hours"]), float(row["peak_window_hours"]))
        park = parking_plan(peak_spaces, int(row["permanent_spaces"]), int(row["temporary_spaces"]))
        shuttle_vehicles, driver_schedule = _shuttle_schedule(visitors, row)
        entry_schedule = _staff_schedule(visitors, BLOCK_SHARES, float(row["entry_service_rate"]), float(row["utilisation_target"]), 1, multiplier=1 / float(row["average_party_size"]))
        parking_schedule = _staff_schedule(visitors, BLOCK_SHARES, float(row["parking_service_rate"]), float(row["utilisation_target"]), 1, multiplier=float(row["car_share"]) / float(row["average_party_size"]))
        records.append({
            "date": row["date"], "region_code": row["region_code"], "visitor_estimate_q50": visitors,
            "demand_scenario": row["demand_scenario"], "data_basis": "derived_planning_value",
            "parking_capacity_basis": row.get("parking_capacity_basis", "scenario_parameter"),
            "permanent_capacity_available": int(row["permanent_spaces"]),
            "temporary_capacity_available": int(row["temporary_spaces"]),
            **park,
            "recommended_shuttle_vehicles": shuttle_vehicles,
            "entry_early_shift": entry_schedule["early_shift"], "entry_middle_shift": entry_schedule["middle_shift"], "entry_late_shift": entry_schedule["late_shift"], "entry_staff_shifts": entry_schedule["staff_shifts"],
            "parking_early_shift": parking_schedule["early_shift"], "parking_middle_shift": parking_schedule["middle_shift"], "parking_late_shift": parking_schedule["late_shift"], "parking_staff_shifts": parking_schedule["staff_shifts"],
            "driver_early_shift": driver_schedule["early_shift"], "driver_middle_shift": driver_schedule["middle_shift"], "driver_late_shift": driver_schedule["late_shift"], "driver_staff_shifts": driver_schedule["staff_shifts"],
            "recommended_total_staff_shifts": entry_schedule["staff_shifts"] + parking_schedule["staff_shifts"] + driver_schedule["staff_shifts"],
            "output_status": "recommended_planning_configuration_not_operating_ledger",
        })
    return pd.DataFrame(records)


def build_all_scenario_plans(forecast: pd.DataFrame, scenarios: pd.DataFrame) -> pd.DataFrame:
    """Apply one scenario row to its matching regional forecast only."""
    if "forecast_quantile" not in scenarios.columns:
        raise ValueError("scenarios must specify forecast_quantile")
    plans: list[pd.DataFrame] = []
    for _, scenario in scenarios.iterrows():
        quantile = scenario["forecast_quantile"]
        if quantile not in forecast.columns:
            raise ValueError(f"forecast is missing quantile column: {quantile}")
        regional_forecast = forecast.loc[forecast["region_code"].eq(scenario["region_code"]), ["date", "region_code", quantile]].rename(columns={quantile: "visitor_estimate_q50"})
        if regional_forecast.empty:
            raise ValueError(f"forecast has no records for {scenario['region_code']}")
        plan = build_daily_absolute_plan(regional_forecast, pd.DataFrame([scenario]))
        plan["forecast_quantile"] = quantile
        plans.append(plan)
    return pd.concat(plans, ignore_index=True)
