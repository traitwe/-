import pandas as pd

from src.analysis.vot_outputs import attach_vot_costs


def test_attach_vot_costs_adds_low_central_high_currency_estimates():
    plan = pd.DataFrame(
        {
            "visitor_estimate_q50": [100.0],
            "parking_unserved_spaces": [10.0],
            "average_party_size": [2.0],
            "transfer_share": [0.5],
            "shuttle_shortfall": [1.0],
            "recommended_shuttle_vehicles": [10.0],
            "staff_shortfall": [2.0],
            "recommended_total_staff_shifts": [10.0],
        }
    )
    scenarios = pd.DataFrame(
        {
            "vot_scenario": ["low", "central", "high"],
            "vot_cny_per_hour": [10.0, 20.0, 30.0],
            "parking_delay_hours": [0.5, 0.5, 0.5],
            "shuttle_delay_hours": [0.25, 0.25, 0.25],
            "staff_delay_hours": [0.1, 0.1, 0.1],
        }
    )

    result = attach_vot_costs(plan, scenarios)

    assert result.loc[0, "traveler_time_loss_hours"] == 13.25
    assert result.loc[0, "traveler_time_loss_cny_low"] == 132.5
    assert result.loc[0, "traveler_time_loss_cny_central"] == 265.0
    assert result.loc[0, "traveler_time_loss_cny_high"] == 397.5
