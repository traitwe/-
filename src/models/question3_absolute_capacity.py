"""Transparent capacity primitives for Question 3 planning recommendations.

The functions return *recommended planning quantities*.  They must not be
interpreted as a reconstruction of a scenic area's actual operating ledger.
"""

from __future__ import annotations

import math


def _positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def peak_parking_demand(
    daily_visitors: float,
    car_share: float,
    average_party_size: float,
    peak_arrival_share: float,
    dwell_hours: float,
    peak_window_hours: float,
) -> int:
    """Estimate concurrent parking spaces in the peak window.

    ``daily_visitors`` is persons/day; car share is the fraction arriving by
    private car; party size is persons/vehicle.  The result is vehicles/spaces.
    """
    if daily_visitors < 0:
        raise ValueError("daily_visitors must be non-negative")
    if not 0 <= car_share <= 1:
        raise ValueError("car_share must be in [0, 1]")
    if not 0 <= peak_arrival_share <= 1:
        raise ValueError("peak_arrival_share must be in [0, 1]")
    _positive(average_party_size, "average_party_size")
    _positive(dwell_hours, "dwell_hours")
    _positive(peak_window_hours, "peak_window_hours")
    concurrent = daily_visitors * car_share / average_party_size * peak_arrival_share * dwell_hours / peak_window_hours
    return math.ceil(concurrent)


def parking_plan(peak_spaces: int, permanent_spaces: int, temporary_spaces: int = 0) -> dict[str, int | str]:
    """Allocate a peak parking requirement to permanent and temporary spaces."""
    for value, name in ((peak_spaces, "peak_spaces"), (permanent_spaces, "permanent_spaces"), (temporary_spaces, "temporary_spaces")):
        if value < 0:
            raise ValueError(f"{name} must be non-negative")
    permanent_open = min(peak_spaces, permanent_spaces)
    remaining = peak_spaces - permanent_open
    temporary_open = min(remaining, temporary_spaces)
    unserved = remaining - temporary_open
    if unserved:
        status = "parking_overflow_unserved"
    elif temporary_open:
        status = "temporary_capacity_required"
    else:
        status = "permanent_capacity_sufficient"
    return {
        "peak_spaces_required": peak_spaces,
        "permanent_spaces_open": permanent_open,
        "temporary_spaces_open": temporary_open,
        "unserved_spaces": unserved,
        "capacity_status": status,
    }


def required_shuttle_vehicles(
    passenger_demand_per_hour: float,
    seats_per_vehicle: int,
    load_factor: float,
    round_trip_minutes: float,
    maximum_headway_minutes: float,
) -> int:
    """Return the simultaneous fleet needed for demand and headway targets."""
    if passenger_demand_per_hour < 0:
        raise ValueError("passenger_demand_per_hour must be non-negative")
    _positive(seats_per_vehicle, "seats_per_vehicle")
    if not 0 < load_factor <= 1:
        raise ValueError("load_factor must be in (0, 1]")
    _positive(round_trip_minutes, "round_trip_minutes")
    _positive(maximum_headway_minutes, "maximum_headway_minutes")
    demand_fleet = math.ceil(passenger_demand_per_hour * round_trip_minutes / (60 * seats_per_vehicle * load_factor))
    headway_fleet = math.ceil(round_trip_minutes / maximum_headway_minutes)
    return max(demand_fleet, headway_fleet)


def minimum_staff(
    arrival_rate_per_hour: float,
    service_rate_per_staff_hour: float,
    utilisation_target: float,
    minimum_people: int = 1,
) -> int:
    """Return staff needed to keep workload at or below the utilisation target."""
    if arrival_rate_per_hour < 0:
        raise ValueError("arrival_rate_per_hour must be non-negative")
    _positive(service_rate_per_staff_hour, "service_rate_per_staff_hour")
    if not 0 < utilisation_target <= 1:
        raise ValueError("utilisation_target must be in (0, 1]")
    if minimum_people < 0:
        raise ValueError("minimum_people must be non-negative")
    required = math.ceil(arrival_rate_per_hour / (service_rate_per_staff_hour * utilisation_target))
    return max(required, minimum_people)
