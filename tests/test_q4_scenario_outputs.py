import pandas as pd

from src.analysis.q4_scenario_outputs import compare_baseline_with_reoptimisation


def test_scenario_comparison_reports_baseline_gap_and_reoptimised_plan():
    baseline = pd.DataFrame({"date": ["2026-08-08"], "region_code": ["BDH"], "visitor_estimate_q50": [10_000], "peak_spaces_required": [500], "permanent_capacity_available": [400], "temporary_capacity_available": [200], "recommended_shuttle_vehicles": [2], "recommended_total_staff_shifts": [5]})
    shocked = pd.DataFrame({"date": ["2026-08-08"], "region_code": ["BDH"], "scenario_visitors": [15_000]})
    scenario = {"car_share": 0.5, "average_party_size": 3.0, "peak_arrival_share": 0.5, "dwell_hours": 3.0, "peak_window_hours": 4.0, "transfer_share": 0.1, "seats_per_vehicle": 40, "load_factor": 0.8, "round_trip_minutes": 40, "maximum_headway_minutes": 15, "entry_service_rate": 40, "parking_service_rate": 50, "utilisation_target": 0.8}

    result = compare_baseline_with_reoptimisation(baseline, shocked, scenario)

    assert result.loc[0, "baseline_service_gap"] > 0
    assert result.loc[0, "reoptimized_shuttle_vehicles"] >= result.loc[0, "baseline_shuttle_vehicles"]
    assert result.loc[0, "scenario_status"] == "counterfactual_planning_simulation"
