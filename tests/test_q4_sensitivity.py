import pandas as pd

from src.analysis.q4_scenario_outputs import one_at_a_time_sensitivity


def test_sensitivity_recomputes_low_and_high_parameter_effects():
    baseline = pd.DataFrame({"date": ["2026-08-07", "2026-08-08", "2026-08-09"], "region_code": ["BDH"] * 3, "visitor_estimate_q50": [10_000] * 3, "permanent_capacity_available": [400] * 3, "temporary_capacity_available": [200] * 3, "recommended_shuttle_vehicles": [2] * 3, "recommended_total_staff_shifts": [5] * 3})
    scenario = {"car_share": 0.5, "average_party_size": 3.0, "peak_arrival_share": 0.5, "dwell_hours": 3.0, "peak_window_hours": 4.0, "transfer_share": 0.1, "seats_per_vehicle": 40, "load_factor": 0.8, "round_trip_minutes": 40, "maximum_headway_minutes": 15, "entry_service_rate": 40, "parking_service_rate": 50, "utilisation_target": 0.8}

    result = one_at_a_time_sensitivity(baseline, scenario, "2026-08-08", {"event_intensity": (0.2, 0.6), "car_share": (0.3, 0.7)})

    assert set(result["parameter"]) == {"event_intensity", "car_share"}
    assert result["sensitivity_effect"].gt(0).all()


def test_cost_experience_penalty_changes_reoptimised_resource_choice():
    from src.analysis.q4_scenario_outputs import compare_baseline_with_reoptimisation
    baseline = pd.DataFrame({"date": ["2026-08-08"], "region_code": ["BDH"], "visitor_estimate_q50": [10_000], "permanent_capacity_available": [400], "temporary_capacity_available": [200], "recommended_shuttle_vehicles": [2], "recommended_total_staff_shifts": [5]})
    shocked = pd.DataFrame({"date": ["2026-08-08"], "region_code": ["BDH"], "scenario_visitors": [15_000]})
    scenario = {"car_share": 0.5, "average_party_size": 3.0, "peak_arrival_share": 0.5, "dwell_hours": 3.0, "peak_window_hours": 4.0, "transfer_share": 0.1, "seats_per_vehicle": 40, "load_factor": 0.8, "round_trip_minutes": 40, "maximum_headway_minutes": 15, "entry_service_rate": 40, "parking_service_rate": 50, "utilisation_target": 0.8}
    low = compare_baseline_with_reoptimisation(baseline, shocked, scenario, risk_penalty=1.0)
    high = compare_baseline_with_reoptimisation(baseline, shocked, scenario, risk_penalty=8.0)
    assert high.loc[0, "reoptimized_relative_cost"] >= low.loc[0, "reoptimized_relative_cost"]
