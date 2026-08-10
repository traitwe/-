"""Bridge Q4 counterfactual shocks to Q3 resource planning models."""

from __future__ import annotations

import pandas as pd

from src.analysis.q3_absolute_resource_outputs import build_daily_absolute_plan
from src.models.question3_absolute_optimizer import optimize_absolute_resources
from src.models.question4_scenarios import apply_event_pulse, service_gap_index


def compare_baseline_with_reoptimisation(baseline: pd.DataFrame, shocked: pd.DataFrame, scenario: dict[str, float], risk_penalty: float = 4.0) -> pd.DataFrame:
    """Compare fixed baseline resources against a shocked, re-optimised plan."""
    merged = baseline.merge(shocked.loc[:, ["date", "region_code", "scenario_visitors"]], on=["date", "region_code"], validate="one_to_one")
    scenario_row = {**scenario, "region_code": merged.loc[0, "region_code"], "demand_scenario": "q4_counterfactual", "permanent_spaces": int(merged.loc[0, "permanent_capacity_available"]), "temporary_spaces": int(merged.loc[0, "temporary_capacity_available"])}
    demand = build_daily_absolute_plan(merged.loc[:, ["date", "region_code", "scenario_visitors"]].rename(columns={"scenario_visitors": "visitor_estimate_q50"}), pd.DataFrame([scenario_row]))
    rows = []
    for index, base in merged.iterrows():
        required = demand.iloc[index]
        gap = service_gap_index(float(required["peak_spaces_required"]), float(base["permanent_capacity_available"] + base["temporary_capacity_available"]), float(required["recommended_shuttle_vehicles"]), float(base["recommended_shuttle_vehicles"]), float(required["recommended_total_staff_shifts"]), float(base["recommended_total_staff_shifts"]))
        chosen, _ = optimize_absolute_resources(int(required["peak_spaces_required"]), int(base["permanent_capacity_available"]), int(base["temporary_capacity_available"]), int(required["recommended_shuttle_vehicles"]), int(required["recommended_total_staff_shifts"]), risk_penalty)
        rows.append({"date": base["date"], "region_code": base["region_code"], "baseline_visitors": base["visitor_estimate_q50"], "scenario_visitors": base["scenario_visitors"], "baseline_service_gap": gap, "baseline_shuttle_vehicles": base["recommended_shuttle_vehicles"], "reoptimized_shuttle_vehicles": chosen["selected_shuttle_vehicles"], "baseline_staff_shifts": base["recommended_total_staff_shifts"], "reoptimized_staff_shifts": chosen["selected_staff_shifts"], "reoptimized_temporary_spaces": chosen["selected_temporary_spaces"], "reoptimized_relative_cost": chosen["relative_operating_cost"], "reoptimized_experience_loss": chosen["standardized_experience_loss"], "scenario_status": "counterfactual_planning_simulation"})
    return pd.DataFrame(rows)


def one_at_a_time_sensitivity(baseline: pd.DataFrame, scenario: dict[str, float], event_date: str, ranges: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Recompute the event scenario at low/high values for each named parameter."""
    records = []
    for parameter, (low, high) in ranges.items():
        gaps = []
        for value in (low, high):
            adjusted = dict(scenario)
            intensity = 0.4
            if parameter == "event_intensity":
                intensity = value
            else:
                adjusted[parameter] = value
            shocked = apply_event_pulse(baseline, "visitor_estimate_q50", event_date, intensity)
            comparison = compare_baseline_with_reoptimisation(baseline, shocked, adjusted)
            gaps.append(float(comparison["baseline_service_gap"].max()))
        records.append({"parameter": parameter, "low_value": low, "high_value": high, "low_max_service_gap": gaps[0], "high_max_service_gap": gaps[1], "sensitivity_effect": abs(gaps[1] - gaps[0])})
    result = pd.DataFrame(records)
    result["ranking"] = result["sensitivity_effect"].rank(method="dense", ascending=False).astype(int)
    return result.sort_values(["ranking", "parameter"]).reset_index(drop=True)
