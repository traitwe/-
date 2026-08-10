"""Build reconstructed Question 3 relative service-tier outputs."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.question3_relative_tiers import beidaihe_summer_trigger, classify_relative_pressure
from src.models.question3_relative_optimizer import optimize_relative_tiers

forecast = pd.read_csv(ROOT / "outputs/question2_analysis/q2_region_daily_forecast_2026.csv", encoding="utf-8-sig")
calendar = pd.read_csv(ROOT / "data/clean/calendar_2015_2026.csv", encoding="utf-8-sig")
calendar = calendar.loc[:, ["date", "\u662f\u5426\u5468\u672b", "\u662f\u5426\u6cd5\u5b9a\u653e\u5047"]].rename(columns={"\u662f\u5426\u5468\u672b": "is_weekend", "\u662f\u5426\u6cd5\u5b9a\u653e\u5047": "is_holiday"})
daily = classify_relative_pressure(forecast).merge(calendar, on="date", how="left")
daily["base_service_tier"] = daily["pressure_tier"].map({"low": "normal", "normal": "normal", "high": "surge", "extreme": "surge"})
bdh = daily["region_code"].eq("BDH") & pd.to_datetime(daily["date"]).dt.month.isin([7, 8])
daily.loc[bdh, "base_service_tier"] = daily.loc[bdh].apply(lambda x: beidaihe_summer_trigger(x["relative_pressure"], bool(x["is_weekend"]), bool(x["is_holiday"])), axis=1)
actions = {"normal": "maintain_normal_staffing__15min_summer_shuttle_baseline_if_operating", "surge": "intensify_parking_guidance__open_surge_window_security_teams__prepare_outer_transfer", "emergency": "activate_outer_transfer__issue_parking_saturation_alert__emergency_coordination_and_staggered_entry"}
daily["recommended_action"] = daily["base_service_tier"].map(actions)
budget_by_tier = {"normal": 1.2, "surge": 3.0, "emergency": 5.5}
anchor_support = {"BDH": {"shuttle": 0.20}, "HGA": {"parking_guidance": 0.30}, "SHG": {"parking_guidance": 0.20}}
optimized = daily.apply(lambda row: optimize_relative_tiers(float(row["relative_pressure"]), risk_penalty=4.0, budget=budget_by_tier[row["base_service_tier"]], anchor_support=anchor_support[row["region_code"]])[0], axis=1).apply(pd.Series)
daily = pd.concat([daily, optimized.drop(columns=["target_tier"], errors="ignore")], axis=1)
daily["output_status"] = "relative_service_intensity_not_exact_resource_count"
output = ROOT / "outputs/question3_analysis"; output.mkdir(parents=True, exist_ok=True)
daily.to_csv(output / "q3_daily_relative_service_tiers_2026.csv", index=False, encoding="utf-8-sig")
summary = daily.groupby(["region_code", "base_service_tier"], as_index=False).size().rename(columns={"size": "days"})
summary.to_csv(output / "q3_service_tier_summary_2026.csv", index=False, encoding="utf-8-sig")
daily.groupby(["region_code", "base_service_tier"], as_index=False)[["staff_tier", "entry_tier", "parking_guidance_tier", "shuttle_tier", "relative_cost", "standardized_service_risk"]].mean().to_csv(output / "q3_multiresource_optimization_summary_2026.csv", index=False, encoding="utf-8-sig")

tier_columns = ["staff_tier", "entry_tier", "parking_guidance_tier", "shuttle_tier"]
static = daily.groupby("region_code", as_index=False)[tier_columns].max()
cost_weight = {"staff_tier": 1.0, "entry_tier": 0.8, "parking_guidance_tier": 0.6, "shuttle_tier": 1.2}
static["static_daily_relative_cost"] = sum(static[column] * weight for column, weight in cost_weight.items())
dynamic = daily.groupby("region_code", as_index=False).agg(dynamic_daily_relative_cost=("relative_cost", "mean"), dynamic_mean_service_risk=("standardized_service_risk", "mean"))
comparison = static.merge(dynamic, on="region_code")
comparison["annual_relative_cost_static"] = comparison["static_daily_relative_cost"] * 365
comparison["annual_relative_cost_dynamic"] = comparison["dynamic_daily_relative_cost"] * 365
comparison["relative_cost_reduction"] = 1 - comparison["annual_relative_cost_dynamic"] / comparison["annual_relative_cost_static"]
comparison["comparison_note"] = "relative_cost_scenario_not_currency__same_service_risk_definition"
comparison.to_csv(output / "q3_static_vs_dynamic_comparison_2026.csv", index=False, encoding="utf-8-sig")

bdh = daily.loc[(daily["region_code"].eq("BDH")) & (pd.to_datetime(daily["date"]).dt.month.isin([7, 8])) & daily["base_service_tier"].isin(["surge", "emergency"])]
bdh.loc[:, ["date", "relative_pressure", "base_service_tier", "staff_tier", "entry_tier", "parking_guidance_tier", "shuttle_tier", "recommended_action"]].to_csv(output / "q3_beidaihe_summer_temporary_plan_2026.csv", index=False, encoding="utf-8-sig")

pareto_rows = []
for penalty in [1.0, 2.0, 4.0, 6.0, 8.0]:
    scenario = daily.apply(lambda row: optimize_relative_tiers(float(row["relative_pressure"]), risk_penalty=penalty, budget=budget_by_tier[row["base_service_tier"]], anchor_support=anchor_support[row["region_code"]])[0], axis=1).apply(pd.Series)
    pareto_rows.append({"risk_penalty": penalty, "mean_relative_cost": scenario["relative_cost"].mean(), "mean_standardized_service_risk": scenario["standardized_service_risk"].mean()})
pareto = pd.DataFrame(pareto_rows).sort_values("mean_relative_cost")
pareto["is_nondominated"] = ~pareto.apply(lambda row: ((pareto["mean_relative_cost"] <= row["mean_relative_cost"]) & (pareto["mean_standardized_service_risk"] <= row["mean_standardized_service_risk"]) & ((pareto["mean_relative_cost"] < row["mean_relative_cost"]) | (pareto["mean_standardized_service_risk"] < row["mean_standardized_service_risk"]))).any(), axis=1)
pareto.to_csv(output / "q3_cost_experience_pareto_2026.csv", index=False, encoding="utf-8-sig")
