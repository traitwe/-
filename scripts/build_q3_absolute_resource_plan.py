"""Build absolute, scenario-based Question 3 resource planning outputs."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.q3_absolute_resource_outputs import build_all_scenario_plans
from src.models.question3_absolute_optimizer import optimize_absolute_resources


def season_from_month(month: int) -> str:
    if month in (7, 8):
        return "peak_summer"
    if month in (4, 5, 6, 9, 10):
        return "shoulder"
    return "low_season"


forecast = pd.read_csv(ROOT / "outputs/question2_analysis/q2_region_daily_forecast_2026.csv", encoding="utf-8-sig")
scenarios = pd.read_csv(ROOT / "data/model_input/q3_absolute_planning_scenarios.csv", encoding="utf-8-sig")
daily = build_all_scenario_plans(forecast, scenarios)
daily = daily.merge(scenarios.loc[:, ["region_code", "demand_scenario", "parameter_basis", "parameter_note"]], on=["region_code", "demand_scenario"], how="left", validate="many_to_one")
daily["season"] = pd.to_datetime(daily["date"]).dt.month.map(season_from_month)
output = ROOT / "outputs/question3_analysis"
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
