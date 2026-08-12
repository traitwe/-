# question4_core.py: 以下函数从完整运行程序的真实实现原样摘取。

# 完整输入、输出与辅助函数见支撑材料中的 question*_model.py。

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
