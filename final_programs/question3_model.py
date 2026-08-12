"""Runnable final program for one modelling question."""
from pathlib import Path
from common_runtime import _export_submission, prepare_submission_runtime


"""Transparent capacity primitives for Question 3 planning recommendations.

The functions return *recommended planning quantities*.  They must not be
interpreted as a reconstruction of a scenic area's actual operating ledger.
"""



import math


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def peak_parking_demand(
    daily_visitors: float,
    car_share: float,
    average_party_size: float,
    peak_arrival_share: float,
    dwell_hours: float,
    peak_window_hours: float,
) -> int:
    """Estimate concurrent parking spaces in the peak window.

    ``daily_visitors`` is persons/day; car share is the fraction arriving by
    private car; party size is persons/vehicle.  The result is vehicles/spaces.
    """
    if daily_visitors < 0:
        raise ValueError("daily_visitors must be non-negative")
    if not 0 <= car_share <= 1:
        raise ValueError("car_share must be in [0, 1]")
    if not 0 <= peak_arrival_share <= 1:
        raise ValueError("peak_arrival_share must be in [0, 1]")
    _positive(average_party_size, "average_party_size")
    _positive(dwell_hours, "dwell_hours")
    _positive(peak_window_hours, "peak_window_hours")
    concurrent = daily_visitors * car_share / average_party_size * peak_arrival_share * dwell_hours / peak_window_hours
    return math.ceil(concurrent)


def parking_plan(peak_spaces: int, permanent_spaces: int, temporary_spaces: int = 0) -> dict[str, int | str]:
    """Allocate a peak parking requirement to permanent and temporary spaces."""
    for value, name in ((peak_spaces, "peak_spaces"), (permanent_spaces, "permanent_spaces"), (temporary_spaces, "temporary_spaces")):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    permanent_open = min(peak_spaces, permanent_spaces)
    remaining = peak_spaces - permanent_open
    temporary_open = min(remaining, temporary_spaces)
    unserved = remaining - temporary_open
    if unserved:
        status = "parking_overflow_unserved"
    elif temporary_open:
        status = "temporary_capacity_required"
    else:
        status = "permanent_capacity_sufficient"
    return {
        "peak_spaces_required": peak_spaces,
        "permanent_spaces_open": permanent_open,
        "temporary_spaces_open": temporary_open,
        "unserved_spaces": unserved,
        "capacity_status": status,
    }


def required_shuttle_vehicles(
    passenger_demand_per_hour: float,
    seats_per_vehicle: int,
    load_factor: float,
    round_trip_minutes: float,
    maximum_headway_minutes: float,
) -> int:
    """Return the simultaneous fleet needed for demand and headway targets."""
    if passenger_demand_per_hour < 0:
        raise ValueError("passenger_demand_per_hour must be non-negative")
    _positive(seats_per_vehicle, "seats_per_vehicle")
    if not 0 < load_factor <= 1:
        raise ValueError("load_factor must be in (0, 1]")
    _positive(round_trip_minutes, "round_trip_minutes")
    _positive(maximum_headway_minutes, "maximum_headway_minutes")
    demand_fleet = math.ceil(passenger_demand_per_hour * round_trip_minutes / (60 * seats_per_vehicle * load_factor))
    headway_fleet = math.ceil(round_trip_minutes / maximum_headway_minutes)
    return max(demand_fleet, headway_fleet)


def minimum_staff(
    arrival_rate_per_hour: float,
    service_rate_per_staff_hour: float,
    utilisation_target: float,
    minimum_people: int = 1,
) -> int:
    """Return staff needed to keep workload at or below the utilisation target."""
    if arrival_rate_per_hour < 0:
        raise ValueError("arrival_rate_per_hour must be non-negative")
    _positive(service_rate_per_staff_hour, "service_rate_per_staff_hour")
    if not 0 < utilisation_target <= 1:
        raise ValueError("utilisation_target must be in (0, 1]")
    if minimum_people < 0:
        raise ValueError("minimum_people must be non-negative")
    required = math.ceil(arrival_rate_per_hour / (service_rate_per_staff_hour * utilisation_target))
    return max(required, minimum_people)


"""Small integer scheduler for three overlapping operational shifts."""




def optimize_three_shifts(required_people_by_block: list[int]) -> dict[str, int]:
    """Minimise staff-shifts while covering early/mid/late demand.

    The early shift covers blocks 1–2, the middle shift blocks 1–3 and the
    late shift blocks 2–3.  These are staff-shift assignments, not a record of
    currently employed people.
    """
    if len(required_people_by_block) != 3:
        raise ValueError("required_people_by_block must contain early, middle and late demand")
    if any(int(value) != value or value < 0 for value in required_people_by_block):
        raise ValueError("required_people_by_block must be non-negative integers")
    early_required, middle_required, late_required = required_people_by_block
    upper = max(required_people_by_block)
    candidates: list[dict[str, int]] = []
    for middle_shift in range(upper + 1):
        early_shift = max(early_required - middle_shift, 0)
        late_shift = max(late_required - middle_shift, 0)
        total = early_shift + middle_shift + late_shift
        if total < middle_required:
            late_shift += middle_required - total
            total = middle_required
        coverage = [early_shift + middle_shift, total, middle_shift + late_shift]
        candidates.append({
            "early_shift": early_shift,
            "middle_shift": middle_shift,
            "late_shift": late_shift,
            "staff_shifts": total,
            "overstaffing_person_blocks": sum(provided - required for provided, required in zip(coverage, required_people_by_block)),
        })
    return min(candidates, key=lambda row: (row["staff_shifts"], row["overstaffing_person_blocks"], row["middle_shift"], row["early_shift"]))


"""Cost--experience optimisation over absolute Q3 planning quantities."""



from itertools import product
import math

import pandas as pd


def optimize_absolute_resources(
    parking_required: int,
    permanent_spaces: int,
    temporary_capacity: int,
    shuttle_required: int,
    staff_required: int,
    risk_penalty: float = 4.0,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Enumerate six coverage levels for temporary parking, shuttles and staff.

    Costs are relative scenario units: temporary-space activation, vehicle
    deployment, and staff-shift deployment each equal one at full coverage.
    Experience loss is the weighted fraction of unmet parking, shuttle and
    staffing need.  It is not a measured satisfaction score.
    """
    values = (parking_required, permanent_spaces, temporary_capacity, shuttle_required, staff_required)
    if any(int(value) != value or value < 0 for value in values) or risk_penalty < 0:
        raise ValueError("resource requirements and capacities must be non-negative integers; risk_penalty non-negative")
    levels = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    records = []
    for parking_level, shuttle_level, staff_level in product(levels, repeat=3):
        temporary = math.ceil(temporary_capacity * parking_level)
        shuttles = math.ceil(shuttle_required * shuttle_level)
        staff = math.ceil(staff_required * staff_level)
        parking_unserved = max(parking_required - permanent_spaces - temporary, 0)
        shuttle_shortfall = max(shuttle_required - shuttles, 0)
        staff_shortfall = max(staff_required - staff, 0)
        parking_loss = parking_unserved / max(parking_required, 1)
        shuttle_loss = shuttle_shortfall / max(shuttle_required, 1)
        staff_loss = staff_shortfall / max(staff_required, 1)
        experience_loss = 0.9 * parking_loss + 1.2 * shuttle_loss + staff_loss
        relative_cost = (
            0.6 * temporary / max(temporary_capacity, 1)
            + 1.2 * shuttles / max(shuttle_required, 1)
            + staff / max(staff_required, 1)
        )
        records.append({
            "selected_temporary_spaces": temporary,
            "selected_shuttle_vehicles": shuttles,
            "selected_staff_shifts": staff,
            "parking_unserved_spaces": parking_unserved,
            "shuttle_shortfall": shuttle_shortfall,
            "staff_shortfall": staff_shortfall,
            "standardized_experience_loss": experience_loss,
            "relative_operating_cost": relative_cost,
            "objective": relative_cost + risk_penalty * experience_loss,
        })
    diagnostics = pd.DataFrame(records).sort_values(["objective", "relative_operating_cost", "standardized_experience_loss"]).reset_index(drop=True)
    return diagnostics.iloc[0].to_dict(), diagnostics


"""Transparent visitor time-loss valuation for Q3/Q4 planning scenarios."""




def _non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _share(value: float, name: str) -> float:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def estimate_traveler_time_loss(
    *,
    daily_visitors: float,
    parking_unserved_spaces: float,
    average_party_size: float,
    parking_delay_hours: float,
    transfer_share: float,
    shuttle_shortfall_fraction: float,
    shuttle_delay_hours: float,
    staff_shortfall_fraction: float,
    staff_delay_hours: float,
    vot_cny_per_hour: float,
) -> dict[str, float]:
    """Estimate visitor time loss and its VOT-valued currency equivalent.

    The result measures scenario traveller inconvenience only.  It must not be
    interpreted as an operator cash ledger, compensation amount, or survey-based
    willingness-to-pay estimate.
    """
    visitors = _non_negative(daily_visitors, "daily_visitors")
    unserved_spaces = _non_negative(parking_unserved_spaces, "parking_unserved_spaces")
    party_size = float(average_party_size)
    if party_size <= 0:
        raise ValueError("average_party_size must be positive")
    parking_delay = _non_negative(parking_delay_hours, "parking_delay_hours")
    transfer = _share(transfer_share, "transfer_share")
    shuttle_shortfall = _share(shuttle_shortfall_fraction, "shuttle_shortfall_fraction")
    shuttle_delay = _non_negative(shuttle_delay_hours, "shuttle_delay_hours")
    staff_shortfall = _share(staff_shortfall_fraction, "staff_shortfall_fraction")
    staff_delay = _non_negative(staff_delay_hours, "staff_delay_hours")
    vot = _non_negative(vot_cny_per_hour, "vot_cny_per_hour")

    parking_hours = unserved_spaces * party_size * parking_delay
    shuttle_hours = visitors * transfer * shuttle_shortfall * shuttle_delay
    staff_hours = visitors * staff_shortfall * staff_delay
    total_hours = parking_hours + shuttle_hours + staff_hours
    return {
        "parking_time_loss_hours": parking_hours,
        "shuttle_time_loss_hours": shuttle_hours,
        "staff_time_loss_hours": staff_hours,
        "traveler_time_loss_hours": total_hours,
        "traveler_time_loss_cny": total_hours * vot,
    }


"""Create transparent absolute resource planning values for Question 3."""



import math

import pandas as pd





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
            "parking_transfer_vehicle_demand": int(park["unserved_spaces"]),
            "parking_transfer_visitor_demand": int(math.ceil(park["unserved_spaces"] * float(row["average_party_size"]))),
            "parking_management_action": "outer_park_and_ride_or_timed_reservation" if park["unserved_spaces"] > 0 else "on_site_capacity_sufficient",
            "maximum_headway_minutes": int(row["maximum_headway_minutes"]),
            "operational_anchor_id": row.get("operational_anchor_id", "no_direct_operational_anchor"),
            "headway_evidence_scope": row.get("headway_evidence_scope", "scenario_parameter_without_direct_operational_anchor"),
            "headway_parameter_role": "service_level_planning_constraint__not_actual_fleet_or_trip_ledger",
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


"""Attach transparent VOT-valued traveller time-loss diagnostics to plans."""



import pandas as pd




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

if __name__ == "__main__":
    prepare_submission_runtime(Path(__file__).resolve().parents[1])


"""Build absolute, scenario-based Question 3 resource planning outputs."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]







def season_from_month(month: int) -> str:
    if month in (7, 8):
        return "peak_summer"
    if month in (4, 5, 6, 9, 10):
        return "shoulder"
    return "low_season"


forecast = pd.read_csv(ROOT / "outputs/runtime/question2_analysis/q2_region_daily_forecast_2026.csv", encoding="utf-8-sig")
scenarios = pd.read_csv(ROOT / "data/runtime/model_input/q3_absolute_planning_scenarios.csv", encoding="utf-8-sig")
vot_scenarios = pd.read_csv(ROOT / "data/runtime/model_input/q3_vot_scenarios.csv", encoding="utf-8-sig")
daily = build_all_scenario_plans(forecast, scenarios)
daily = daily.merge(scenarios.loc[:, ["region_code", "demand_scenario", "parameter_basis", "parameter_note", "average_party_size", "transfer_share"]], on=["region_code", "demand_scenario"], how="left", validate="many_to_one")
daily["season"] = pd.to_datetime(daily["date"]).dt.month.map(season_from_month)
output = ROOT / "outputs/runtime/question3_analysis"
output.mkdir(parents=True, exist_ok=True)
daily.to_csv(output / "q3_absolute_daily_resource_plan_2026.csv", index=False, encoding="utf-8-sig")

def optimize_frame(frame: pd.DataFrame, penalty: float) -> pd.DataFrame:
    selected = frame.apply(
        lambda row: optimize_absolute_resources(
            int(row["peak_spaces_required"]),
            int(row["permanent_capacity_available"]),
            int(row["temporary_capacity_available"]),
            int(row["recommended_shuttle_vehicles"]),
            int(row["recommended_total_staff_shifts"]),
            penalty,
        )[0],
        axis=1,
    ).apply(pd.Series)
    return pd.concat([frame.reset_index(drop=True), selected], axis=1)

optimized_daily = optimize_frame(daily, 4.0)
optimized_daily["risk_penalty"] = 4.0
optimized_daily["optimization_status"] = "relative_cost_and_standardized_experience_loss__not_currency_or_survey_score"
optimized_daily = attach_vot_costs(optimized_daily, vot_scenarios)
optimized_daily.to_csv(output / "q3_absolute_daily_optimized_plan_2026.csv", index=False, encoding="utf-8-sig")

pareto_rows = []
for penalty in [1.0, 2.0, 4.0, 6.0, 8.0]:
    candidate = optimize_frame(daily, penalty)
    pareto_rows.append({"risk_penalty": penalty, "mean_relative_operating_cost": candidate["relative_operating_cost"].mean(), "mean_standardized_experience_loss": candidate["standardized_experience_loss"].mean()})
pareto = pd.DataFrame(pareto_rows).sort_values("mean_relative_operating_cost")
pareto["is_nondominated"] = ~pareto.apply(lambda row: ((pareto["mean_relative_operating_cost"] <= row["mean_relative_operating_cost"]) & (pareto["mean_standardized_experience_loss"] <= row["mean_standardized_experience_loss"]) & ((pareto["mean_relative_operating_cost"] < row["mean_relative_operating_cost"]) | (pareto["mean_standardized_experience_loss"] < row["mean_standardized_experience_loss"]))).any(), axis=1)
pareto.to_csv(output / "q3_absolute_cost_experience_pareto_2026.csv", index=False, encoding="utf-8-sig")

seasonal = daily.groupby(["region_code", "demand_scenario", "forecast_quantile", "season", "parking_capacity_basis"], as_index=False).agg(
    representative_daily_visitors=("visitor_estimate_q50", "median"),
    peak_daily_visitors=("visitor_estimate_q50", "max"),
    max_peak_spaces_required=("peak_spaces_required", "max"),
    max_temporary_spaces_open=("temporary_spaces_open", "max"),
    max_unserved_spaces=("unserved_spaces", "max"),
    max_shuttle_vehicles=("recommended_shuttle_vehicles", "max"),
    max_total_staff_shifts=("recommended_total_staff_shifts", "max"),
    parking_overflow_days=("capacity_status", lambda values: (values == "parking_overflow_unserved").sum()),
)
seasonal["result_status"] = "scenario_planning_value_not_operating_ledger"
seasonal.to_csv(output / "q3_absolute_seasonal_resource_plan_2026.csv", index=False, encoding="utf-8-sig")

bdh_summer = daily.loc[(daily["region_code"] == "BDH") & (pd.to_datetime(daily["date"]).dt.month.isin([7, 8]))].copy()
visitor_trigger = bdh_summer.groupby("demand_scenario")["visitor_estimate_q50"].transform(lambda series: series.quantile(0.75))
bdh_summer["temporary_plan_trigger"] = bdh_summer["visitor_estimate_q50"] >= visitor_trigger
bdh_summer.loc[bdh_summer["capacity_status"] != "permanent_capacity_sufficient", "temporary_plan_trigger"] = True
bdh_summer.loc[bdh_summer["temporary_plan_trigger"], "recommended_action"] = "add_peak_shift_staff__dispatch_planned_shuttle_fleet__issue_parking_guidance"
bdh_summer.loc[~bdh_summer["temporary_plan_trigger"], "recommended_action"] = "maintain_seasonal_baseline"
bdh_summer.to_csv(output / "q3_beidaihe_absolute_summer_temporary_plan_2026.csv", index=False, encoding="utf-8-sig")

sensitivity = daily.groupby(["region_code", "demand_scenario", "forecast_quantile"], as_index=False).agg(
    annual_max_peak_spaces_required=("peak_spaces_required", "max"),
    annual_max_shuttle_vehicles=("recommended_shuttle_vehicles", "max"),
    annual_max_staff_shifts=("recommended_total_staff_shifts", "max"),
    annual_parking_overflow_days=("capacity_status", lambda values: (values == "parking_overflow_unserved").sum()),
)
sensitivity["interpretation"] = "low_central_stress_parameter_and_forecast_sensitivity"
sensitivity.to_csv(output / "q3_absolute_resource_sensitivity_2026.csv", index=False, encoding="utf-8-sig")

# Capacity-relaxation diagnostic: transparent counterfactual, not an asserted facility inventory.
bdh_stress = daily.loc[(daily["region_code"] == "BDH") & (daily["demand_scenario"] == "stress")].copy()
peak = bdh_stress.loc[bdh_stress["unserved_spaces"].idxmax()]
slack_rows = []
for added_capacity in (0, 500, 1000):
    chosen, _ = optimize_absolute_resources(int(peak["peak_spaces_required"]), int(peak["permanent_capacity_available"]), added_capacity, int(peak["recommended_shuttle_vehicles"]), int(peak["recommended_total_staff_shifts"]), 4.0)
    physical_shortage = max(int(peak["peak_spaces_required"]) - int(peak["permanent_capacity_available"]) - added_capacity, 0)
    slack_rows.append({"region_code": "BDH", "demand_scenario": "stress", "reference_date": peak["date"], "added_temporary_capacity_scenario": added_capacity, "peak_spaces_required": int(peak["peak_spaces_required"]), "remaining_unserved_spaces_after_full_activation": physical_shortage, "shortage_reduction_vs_no_added_capacity": int(peak["unserved_spaces"]) - physical_shortage, "unconstrained_optimizer_objective": chosen["objective"], "interpretation": "physical_capacity_relaxation_counterfactual_not_verified_supply_or_selected_plan"})
pd.DataFrame(slack_rows).to_csv(output / "q3_bdh_temporary_capacity_relaxation.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    _export_submission(ROOT, "Q3_daily_resource_plan.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_daily_optimized_plan_2026.csv",
    ])
    _export_submission(ROOT, "Q3_seasonal_recommendations.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_seasonal_resource_plan_2026.csv",
        "outputs/runtime/question3_analysis/q3_beidaihe_absolute_summer_temporary_plan_2026.csv",
    ])
    _export_submission(ROOT, "Q3_sensitivity_summary.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_resource_sensitivity_2026.csv",
        "outputs/runtime/question3_analysis/q3_absolute_cost_experience_pareto_2026.csv",
        "outputs/runtime/question3_analysis/q3_bdh_temporary_capacity_relaxation.csv",
    ])
