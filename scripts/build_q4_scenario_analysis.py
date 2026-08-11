"""Build Question 4 event/rain counterfactual and robustness outputs."""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.analysis.q4_scenario_outputs import compare_baseline_with_reoptimisation, one_at_a_time_sensitivity
from src.models.question4_scenarios import apply_continuous_rain, apply_event_pulse

out = ROOT / "outputs/question4_analysis"; out.mkdir(parents=True, exist_ok=True)
base = pd.read_csv(ROOT / "outputs/question3_analysis/q3_absolute_daily_resource_plan_2026.csv", encoding="utf-8-sig")
params = pd.read_csv(ROOT / "data/model_input/q3_absolute_planning_scenarios.csv", encoding="utf-8-sig")
vot = pd.read_csv(ROOT / "data/model_input/q3_vot_scenarios.csv", encoding="utf-8-sig")

def scenario_dict(region: str) -> dict:
    return params[(params.region_code == region) & (params.demand_scenario == "central")].iloc[0].to_dict()


def central_vot() -> dict:
    return vot[vot.vot_scenario.eq("central")].iloc[0].to_dict()

bdh = base[(base.region_code == "BDH") & (base.demand_scenario == "central")].copy()
event = apply_event_pulse(bdh, "visitor_estimate_q50", "2026-08-08", 0.40)
event_result = compare_baseline_with_reoptimisation(bdh, event, scenario_dict("BDH"), vot_parameters=central_vot())
event_result["scenario_name"] = "beidaihe_summer_cultural_event"
event_result.to_csv(out / "q4_event_counterfactual_2026.csv", index=False, encoding="utf-8-sig")

shg = base[(base.region_code == "SHG") & (base.demand_scenario == "central")].copy()
rain = apply_continuous_rain(shg, "visitor_estimate_q50", "2026-07-15", 5, 0.12)
rain_result = compare_baseline_with_reoptimisation(shg, rain, scenario_dict("SHG"), vot_parameters=central_vot())
rain_result["scenario_name"] = "shanhaiguan_five_day_continuous_rain"
rain_result.to_csv(out / "q4_rain_counterfactual_2026.csv", index=False, encoding="utf-8-sig")

ranking = one_at_a_time_sensitivity(bdh, scenario_dict("BDH"), "2026-08-08", {"event_intensity": (0.20, 0.60), "car_share": (0.40, 0.70), "dwell_hours": (2.0, 4.0), "transfer_share": (0.05, 0.15)})
rain_gaps = []
for attenuation in (0.06, 0.18):
    probe = apply_continuous_rain(shg, "visitor_estimate_q50", "2026-07-15", 5, attenuation)
    comparison = compare_baseline_with_reoptimisation(shg, probe, scenario_dict("SHG"))
    rain_gaps.append(float(comparison[comparison.date.between("2026-07-15", "2026-07-19")].baseline_service_gap.max()))
ranking = pd.concat([ranking, pd.DataFrame([{"parameter": "rain_daily_attenuation", "low_value": 0.06, "high_value": 0.18, "low_max_service_gap": rain_gaps[0], "high_max_service_gap": rain_gaps[1], "sensitivity_effect": abs(rain_gaps[1] - rain_gaps[0])}])], ignore_index=True)
ranking["ranking"] = ranking.sensitivity_effect.rank(method="dense", ascending=False).astype(int)
ranking = ranking.sort_values(["ranking", "parameter"])
lambda_rows = []
for penalty in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    probe = compare_baseline_with_reoptimisation(bdh, event, scenario_dict("BDH"), risk_penalty=penalty)
    window = probe[probe.date.between("2026-08-06", "2026-08-10")]
    lambda_rows.append({"risk_penalty": penalty, "mean_relative_cost": window.reoptimized_relative_cost.mean(), "mean_experience_loss": window.reoptimized_experience_loss.mean(), "max_temporary_spaces": window.reoptimized_temporary_spaces.max(), "max_shuttle_vehicles": window.reoptimized_shuttle_vehicles.max(), "max_staff_shifts": window.reoptimized_staff_shifts.max()})
lambda_table = pd.DataFrame(lambda_rows)
lambda_table.to_csv(out / "q4_lambda_tradeoff_2026.csv", index=False, encoding="utf-8-sig")
ranking.to_csv(out / "q4_robustness_sensitivity_ranking.csv", index=False, encoding="utf-8-sig")

summary = pd.concat([event_result.assign(scenario="event"), rain_result.assign(scenario="rain")]).groupby("scenario", as_index=False).agg(max_baseline_gap=("baseline_service_gap", "max"), mean_reoptimized_cost=("reoptimized_relative_cost", "mean"), mean_reoptimized_loss=("reoptimized_experience_loss", "mean"), max_reoptimized_shuttles=("reoptimized_shuttle_vehicles", "max"), max_reoptimized_staff=("reoptimized_staff_shifts", "max"))
summary.to_csv(out / "q4_baseline_reoptimization_summary_2026.csv", index=False, encoding="utf-8-sig")

def window_summary(frame, start, end, name):
    part = frame[frame.date.between(start, end)]
    return {"scenario": name, "window_start": start, "window_end": end, "days": len(part), "max_baseline_gap": part.baseline_service_gap.max(), "max_baseline_shuttles": part.baseline_shuttle_vehicles.max(), "max_reoptimized_shuttles": part.reoptimized_shuttle_vehicles.max(), "max_baseline_staff": part.baseline_staff_shifts.max(), "max_reoptimized_staff": part.reoptimized_staff_shifts.max(), "max_reoptimized_temporary_spaces": part.reoptimized_temporary_spaces.max(), "mean_reoptimized_relative_cost": part.reoptimized_relative_cost.mean(), "mean_reoptimized_experience_loss": part.reoptimized_experience_loss.mean()}

pd.DataFrame([window_summary(event_result, "2026-08-06", "2026-08-10", "beidaihe_event_window"), window_summary(rain_result, "2026-07-15", "2026-07-19", "shanhaiguan_rain_window")]).to_csv(out / "q4_scenario_window_adjustment_summary.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(8, 4)); plt.bar(ranking.parameter, ranking.sensitivity_effect); plt.xticks(rotation=30, ha="right"); plt.ylabel("service-gap effect"); plt.tight_layout(); plt.savefig(out / "q4_sensitivity_ranking.png", dpi=180); plt.close()
plt.figure(figsize=(8, 4)); plt.plot(event_result.date, event_result.baseline_service_gap, label="event baseline gap"); plt.plot(rain_result.date, rain_result.baseline_service_gap, label="rain baseline gap"); plt.legend(); plt.xticks(rotation=30, ha="right"); plt.ylabel("service gap"); plt.tight_layout(); plt.savefig(out / "q4_scenario_service_gap.png", dpi=180); plt.close()
