import pandas as pd

from src.analysis.q3_absolute_resource_outputs import build_all_scenario_plans, build_daily_absolute_plan


def test_absolute_resource_builder_keeps_scope_and_marks_planning_values():
    forecast = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02"],
            "region_code": ["SHG", "SHG"],
            "visitor_estimate_q50": [10_000, 100_000],
        }
    )
    scenarios = pd.DataFrame(
        {
            "region_code": ["SHG"],
            "demand_scenario": ["central"],
            "car_share": [0.6],
            "average_party_size": [3.0],
            "peak_arrival_share": [0.5],
            "dwell_hours": [3.0],
            "peak_window_hours": [4.0],
            "transfer_share": [0.2],
            "seats_per_vehicle": [40],
            "load_factor": [0.8],
            "round_trip_minutes": [30],
            "maximum_headway_minutes": [15],
            "entry_service_rate": [40],
            "parking_service_rate": [50],
            "utilisation_target": [0.8],
            "permanent_spaces": [2225],
            "temporary_spaces": [1000],
            "parking_capacity_basis": ["verified_node"],
        }
    )

    result = build_daily_absolute_plan(forecast, scenarios)

    assert set(result["data_basis"]) == {"derived_planning_value"}
    assert set(result["parking_capacity_basis"]) == {"verified_node"}
    assert set(result["capacity_status"]) == {"permanent_capacity_sufficient", "parking_overflow_unserved"}
    assert result["recommended_shuttle_vehicles"].ge(1).all()
    assert result["recommended_total_staff_shifts"].ge(1).all()
    overflow = result.loc[result["unserved_spaces"] > 0]
    assert not overflow.empty
    assert overflow["parking_transfer_vehicle_demand"].eq(overflow["unserved_spaces"]).all()
    assert overflow["parking_transfer_visitor_demand"].eq(
        (overflow["unserved_spaces"] * 3.0).astype(int)
    ).all()
    assert set(overflow["parking_management_action"]) == {"outer_park_and_ride_or_timed_reservation"}
    assert not result.astype(str).apply(lambda column: column.str.contains("observed_actual")).any().any()


def test_absolute_resource_builder_returns_monotone_peak_demands_for_higher_visitors():
    forecast = pd.DataFrame(
        {
            "date": ["2026-07-01", "2026-07-02"],
            "region_code": ["SHG", "SHG"],
            "visitor_estimate_q50": [10_000, 20_000],
        }
    )
    scenarios = pd.DataFrame(
        {
            "region_code": ["SHG"], "demand_scenario": ["central"], "car_share": [0.6], "average_party_size": [3.0],
            "peak_arrival_share": [0.5], "dwell_hours": [3.0], "peak_window_hours": [4.0], "transfer_share": [0.2],
            "seats_per_vehicle": [40], "load_factor": [0.8], "round_trip_minutes": [30], "maximum_headway_minutes": [15],
            "entry_service_rate": [40], "parking_service_rate": [50], "utilisation_target": [0.8], "permanent_spaces": [2225], "temporary_spaces": [1000], "parking_capacity_basis": ["verified_node"],
        }
    )

    result = build_daily_absolute_plan(forecast, scenarios).sort_values("date")

    assert result["peak_spaces_required"].is_monotonic_increasing
    assert result["recommended_shuttle_vehicles"].is_monotonic_increasing


def test_all_scenario_builder_filters_forecast_to_each_scenario_region():
    forecast = pd.DataFrame({"date": ["2026-07-01", "2026-07-01"], "region_code": ["BDH", "SHG"], "prediction_q10": [100, 200]})
    scenarios = pd.DataFrame({
        "region_code": ["SHG"], "demand_scenario": ["conservative"], "forecast_quantile": ["prediction_q10"],
        "car_share": [0.5], "average_party_size": [3.0], "peak_arrival_share": [0.5], "dwell_hours": [2.0], "peak_window_hours": [4.0],
        "transfer_share": [0.2], "seats_per_vehicle": [30], "load_factor": [0.8], "round_trip_minutes": [30], "maximum_headway_minutes": [15],
        "entry_service_rate": [40], "parking_service_rate": [50], "utilisation_target": [0.8], "permanent_spaces": [2225], "temporary_spaces": [0],
    })

    result = build_all_scenario_plans(forecast, scenarios)

    assert result["region_code"].tolist() == ["SHG"]
    assert result["visitor_estimate_q50"].tolist() == [200]


def test_entry_staffing_uses_party_level_service_units():
    forecast = pd.DataFrame({"date": ["2026-07-01"], "region_code": ["SHG"], "visitor_estimate_q50": [1_000]})
    scenarios = pd.DataFrame({
        "region_code": ["SHG"], "demand_scenario": ["central"], "car_share": [0.5], "average_party_size": [4.0],
        "peak_arrival_share": [0.5], "dwell_hours": [2.0], "peak_window_hours": [4.0], "transfer_share": [0.05],
        "seats_per_vehicle": [30], "load_factor": [0.8], "round_trip_minutes": [30], "maximum_headway_minutes": [15],
        "entry_service_rate": [40], "parking_service_rate": [50], "utilisation_target": [0.8], "permanent_spaces": [2225], "temporary_spaces": [0],
    })

    result = build_daily_absolute_plan(forecast, scenarios)

    assert result.loc[0, "entry_staff_shifts"] == 1
