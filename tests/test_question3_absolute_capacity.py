import pytest

from src.models.question3_absolute_capacity import (
    minimum_staff,
    parking_plan,
    peak_parking_demand,
    required_shuttle_vehicles,
)


def test_peak_parking_demand_converts_visitors_to_concurrent_vehicles():
    result = peak_parking_demand(
        10_000,
        car_share=0.6,
        average_party_size=3,
        peak_arrival_share=0.5,
        dwell_hours=3,
        peak_window_hours=4,
    )

    assert result == 750


def test_parking_plan_reports_overflow_against_a_verified_node_capacity():
    result = parking_plan(1_500, permanent_spaces=1_272, temporary_spaces=300)

    assert result["permanent_spaces_open"] == 1_272
    assert result["temporary_spaces_open"] == 228
    assert result["unserved_spaces"] == 0
    assert result["capacity_status"] == "temporary_capacity_required"


def test_shuttle_fleet_meets_frequency_and_passenger_capacity():
    assert required_shuttle_vehicles(400, 40, 0.8, 30, 15) == 7


def test_minimum_staff_respects_utilisation_target_and_floor():
    assert minimum_staff(120, service_rate_per_staff_hour=30, utilisation_target=0.8, minimum_people=2) == 5


@pytest.mark.parametrize("bad_share", [-0.1, 1.1])
def test_peak_parking_demand_rejects_invalid_car_share(bad_share):
    with pytest.raises(ValueError, match="car_share"):
        peak_parking_demand(100, bad_share, 3, 0.5, 3, 4)
