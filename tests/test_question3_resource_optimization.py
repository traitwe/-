import pandas as pd

from src.models.question3_resource_optimization import allocate_period_demand, evaluate_resource_plan, select_resource_tier


def test_allocate_period_demand_preserves_daily_visitors_and_converts_parking_to_vehicles():
    result = allocate_period_demand(1000.0, [0.2, 0.5, 0.3], car_share=0.6, persons_per_car=3.0, shuttle_share=0.2)

    assert result["visitor_demand"].sum() == 1000.0
    assert result["parking_vehicle_demand"].sum() == 200.0
    assert result["shuttle_passenger_demand"].sum() == 200.0


def test_resource_plan_reports_separate_unit_shortfalls_without_adding_them():
    demand = pd.DataFrame({"period": ["morning"], "visitor_demand": [300.0], "parking_vehicle_demand": [80.0], "shuttle_passenger_demand": [60.0]})
    plan = {"parking_spaces": 50.0, "parking_turnover": 1.0, "shuttle_headway_minutes": 30.0, "shuttle_seats": 20.0, "shuttle_load_factor": 1.0, "period_hours": 2.0, "entry_capacity": 250.0}

    result = evaluate_resource_plan(demand, plan)

    assert result.loc[0, "parking_shortfall_vehicles"] == 30.0
    assert result.loc[0, "shuttle_shortfall_passengers"] == 0.0
    assert result.loc[0, "entry_shortfall_visitors"] == 50.0
    assert "total_shortfall" not in result.columns


def test_resource_tier_selection_trades_declared_cost_against_normalized_service_risk():
    demand = pd.DataFrame({"period": ["morning"], "visitor_demand": [300.0], "parking_vehicle_demand": [80.0], "shuttle_passenger_demand": [60.0]})
    plans = {
        "normal": {"parking_spaces": 50.0, "parking_turnover": 1.0, "shuttle_headway_minutes": 30.0, "shuttle_seats": 20.0, "shuttle_load_factor": 1.0, "period_hours": 2.0, "entry_capacity": 250.0, "tier_cost": 1.0},
        "surge": {"parking_spaces": 90.0, "parking_turnover": 1.0, "shuttle_headway_minutes": 15.0, "shuttle_seats": 20.0, "shuttle_load_factor": 1.0, "period_hours": 2.0, "entry_capacity": 350.0, "tier_cost": 3.0},
    }

    selected, diagnostics = select_resource_tier(demand, plans, risk_penalty=20.0)

    assert selected == "surge"
    assert diagnostics.loc[diagnostics["tier"].eq("surge"), "objective"].iloc[0] < diagnostics.loc[diagnostics["tier"].eq("normal"), "objective"].iloc[0]
