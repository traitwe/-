"""Runnable final program for one modelling question."""
from pathlib import Path
from common_runtime import _export_submission, prepare_submission_runtime
from question3_model import (attach_vot_costs, build_daily_absolute_plan, optimize_absolute_resources)


"""Counterfactual visitor shocks and baseline-plan adequacy measures for Q4."""



import pandas as pd


def _check_column(frame: pd.DataFrame, column: str) -> None:
    if "date" not in frame.columns or column not in frame.columns:
        raise ValueError("frame must contain date and visitor column")
    if (frame[column] < 0).any():
        raise ValueError("visitor values must be non-negative")


def apply_event_pulse(frame: pd.DataFrame, visitor_column: str, event_date: str, intensity: float) -> pd.DataFrame:
    """Apply a five-day 0.25/0.50/1.00/0.50/0.25 event impulse."""
    _check_column(frame, visitor_column)
    if intensity < 0:
        raise ValueError("intensity must be non-negative")
    result = frame.copy()
    dates = pd.to_datetime(result["date"])
    offset = (dates - pd.Timestamp(event_date)).dt.days
    pulse = offset.map({-2: 0.25, -1: 0.50, 0: 1.00, 1: 0.50, 2: 0.25}).fillna(0.0)
    result["scenario_visitors"] = result[visitor_column] * (1 + intensity * pulse)
    result["scenario_type"] = "cultural_tourism_event"
    return result


def apply_continuous_rain(frame: pd.DataFrame, visitor_column: str, start_date: str, duration_days: int, daily_attenuation: float) -> pd.DataFrame:
    """Apply cumulative visitor attenuation across consecutive rainy days."""
    _check_column(frame, visitor_column)
    if duration_days <= 0 or not 0 <= daily_attenuation <= 1:
        raise ValueError("duration_days must be positive and daily_attenuation in [0, 1]")
    result = frame.copy()
    offset = (pd.to_datetime(result["date"]) - pd.Timestamp(start_date)).dt.days
    rainy_day = offset.where(offset.between(0, duration_days - 1), -1)
    multiplier = 1 - daily_attenuation * (rainy_day + 1)
    multiplier = multiplier.where(rainy_day >= 0, 1.0).clip(lower=0)
    result["scenario_visitors"] = result[visitor_column] * multiplier
    result["scenario_type"] = "continuous_rain"
    return result


def service_gap_index(parking_required: float, parking_available: float, shuttle_required: float, shuttle_available: float, staff_required: float, staff_available: float) -> float:
    """Return the equally weighted mean proportional shortfall across services."""
    values = (parking_required, parking_available, shuttle_required, shuttle_available, staff_required, staff_available)
    if any(value < 0 for value in values):
        raise ValueError("service requirements and availability must be non-negative")
    gaps = [max(required - available, 0) / max(required, 1) for required, available in ((parking_required, parking_available), (shuttle_required, shuttle_available), (staff_required, staff_available))]
    return sum(gaps) / len(gaps)


"""Bridge Q4 counterfactual shocks to Q3 resource planning models."""



import pandas as pd







def compare_baseline_with_reoptimisation(baseline: pd.DataFrame, shocked: pd.DataFrame, scenario: dict[str, float], risk_penalty: float = 4.0, vot_parameters: dict[str, float] | None = None) -> pd.DataFrame:
    """Compare fixed baseline resources against a shocked, re-optimised plan."""
    merged = baseline.merge(shocked.loc[:, ["date", "region_code", "scenario_visitors"]], on=["date", "region_code"], validate="one_to_one")
    scenario_row = {**scenario, "region_code": merged.loc[0, "region_code"], "demand_scenario": "q4_counterfactual", "permanent_spaces": int(merged.loc[0, "permanent_capacity_available"]), "temporary_spaces": int(merged.loc[0, "temporary_capacity_available"])}
    demand = build_daily_absolute_plan(merged.loc[:, ["date", "region_code", "scenario_visitors"]].rename(columns={"scenario_visitors": "visitor_estimate_q50"}), pd.DataFrame([scenario_row]))
    rows = []
    for index, base in merged.iterrows():
        required = demand.iloc[index]
        gap = service_gap_index(float(required["peak_spaces_required"]), float(base["permanent_capacity_available"] + base["temporary_capacity_available"]), float(required["recommended_shuttle_vehicles"]), float(base["recommended_shuttle_vehicles"]), float(required["recommended_total_staff_shifts"]), float(base["recommended_total_staff_shifts"]))
        chosen, _ = optimize_absolute_resources(int(required["peak_spaces_required"]), int(base["permanent_capacity_available"]), int(base["temporary_capacity_available"]), int(required["recommended_shuttle_vehicles"]), int(required["recommended_total_staff_shifts"]), risk_penalty)
        baseline_parking_unserved = max(int(required["peak_spaces_required"]) - int(base["permanent_capacity_available"]) - int(base["temporary_capacity_available"]), 0)
        baseline_shuttle_shortfall = max(int(required["recommended_shuttle_vehicles"]) - int(base["recommended_shuttle_vehicles"]), 0)
        baseline_staff_shortfall = max(int(required["recommended_total_staff_shifts"]) - int(base["recommended_total_staff_shifts"]), 0)
        rows.append({"date": base["date"], "region_code": base["region_code"], "baseline_visitors": base["visitor_estimate_q50"], "scenario_visitors": base["scenario_visitors"], "baseline_service_gap": gap, "baseline_shuttle_vehicles": base["recommended_shuttle_vehicles"], "reoptimized_shuttle_vehicles": chosen["selected_shuttle_vehicles"], "baseline_staff_shifts": base["recommended_total_staff_shifts"], "reoptimized_staff_shifts": chosen["selected_staff_shifts"], "reoptimized_temporary_spaces": chosen["selected_temporary_spaces"], "reoptimized_relative_cost": chosen["relative_operating_cost"], "reoptimized_experience_loss": chosen["standardized_experience_loss"], "parking_unserved_spaces": chosen["parking_unserved_spaces"], "shuttle_shortfall": chosen["shuttle_shortfall"], "staff_shortfall": chosen["staff_shortfall"], "baseline_parking_unserved_spaces": baseline_parking_unserved, "baseline_shuttle_shortfall": baseline_shuttle_shortfall, "baseline_staff_shortfall": baseline_staff_shortfall, "recommended_shuttle_vehicles": required["recommended_shuttle_vehicles"], "recommended_total_staff_shifts": required["recommended_total_staff_shifts"], "average_party_size": scenario["average_party_size"], "transfer_share": scenario["transfer_share"], "visitor_estimate_q50": base["scenario_visitors"], "scenario_status": "counterfactual_planning_simulation"})
    result = pd.DataFrame(rows)
    if vot_parameters is not None:
        parameters = dict(vot_parameters)
        parameters.setdefault("vot_scenario", "central")
        baseline_input = result.assign(
            parking_unserved_spaces=result["baseline_parking_unserved_spaces"],
            shuttle_shortfall=result["baseline_shuttle_shortfall"],
            staff_shortfall=result["baseline_staff_shortfall"],
        )
        baseline_vot = attach_vot_costs(baseline_input, pd.DataFrame([parameters]))
        result = attach_vot_costs(result, pd.DataFrame([parameters]))
        result["baseline_traveler_time_loss_hours"] = baseline_vot["traveler_time_loss_hours"]
        result["baseline_traveler_time_loss_cny"] = baseline_vot["traveler_time_loss_cny_central"]
        result = result.rename(columns={"traveler_time_loss_hours": "reoptimized_traveler_time_loss_hours", "traveler_time_loss_cny_central": "reoptimized_traveler_time_loss_cny"})
    return result


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

if __name__ == "__main__":
    prepare_submission_runtime(Path(__file__).resolve().parents[1])


"""Build Question 4 event/rain counterfactual and robustness outputs."""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]




out = ROOT / "outputs/runtime/question4_analysis"; out.mkdir(parents=True, exist_ok=True)
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
base = pd.read_csv(ROOT / "outputs/runtime/question3_analysis/q3_absolute_daily_resource_plan_2026.csv", encoding="utf-8-sig")
params = pd.read_csv(ROOT / "data/runtime/model_input/q3_absolute_planning_scenarios.csv", encoding="utf-8-sig")
vot = pd.read_csv(ROOT / "data/runtime/model_input/q3_vot_scenarios.csv", encoding="utf-8-sig")

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

parameter_labels = {"transfer_share": "接驳分担率", "event_intensity": "活动强度", "car_share": "私家车比例", "dwell_hours": "停留时长", "rain_daily_attenuation": "单日降雨衰减"}
plt.figure(figsize=(8, 4)); plt.bar(ranking.parameter.map(parameter_labels), ranking.sensitivity_effect); plt.xticks(rotation=25, ha="right"); plt.ylabel("对最大服务缺口的影响"); plt.tight_layout(); plt.savefig(out / "q4_sensitivity_ranking.png", dpi=180); plt.close()
plt.figure(figsize=(8, 4)); plt.plot(event_result.date, event_result.baseline_service_gap, label="活动冲击基线缺口"); plt.plot(rain_result.date, rain_result.baseline_service_gap, label="降雨情景基线缺口"); plt.legend(); plt.xticks(rotation=30, ha="right"); plt.ylabel("服务缺口"); plt.tight_layout(); plt.savefig(out / "q4_scenario_service_gap.png", dpi=180); plt.close()

if __name__ == "__main__":
    _export_submission(ROOT, "Q4_scenario_reoptimization.csv", [
        "outputs/runtime/question4_analysis/q4_event_counterfactual_2026.csv",
        "outputs/runtime/question4_analysis/q4_rain_counterfactual_2026.csv",
    ])
    _export_submission(ROOT, "Q4_robustness_summary.csv", [
        "outputs/runtime/question4_analysis/q4_baseline_reoptimization_summary_2026.csv",
        "outputs/runtime/question4_analysis/q4_robustness_sensitivity_ranking.csv",
        "outputs/runtime/question4_analysis/q4_scenario_window_adjustment_summary.csv",
        "outputs/runtime/question4_analysis/q4_lambda_tradeoff_2026.csv",
    ])
