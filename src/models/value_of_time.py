"""Transparent visitor time-loss valuation for Q3/Q4 planning scenarios."""

from __future__ import annotations


def _non_negative(value: float, name: str) -> float:
    numeric = float(value)
    if numeric < 0:
        raise ValueError(f"{name} must be non-negative")
    return numeric


def _share(value: float, name: str) -> float:
    numeric = float(value)
    if not 0 <= numeric <= 1:
        raise ValueError(f"{name} must be in [0, 1]")
    return numeric


def estimate_traveler_time_loss(
    *,
    daily_visitors: float,
    parking_unserved_spaces: float,
    average_party_size: float,
    parking_delay_hours: float,
    transfer_share: float,
    shuttle_shortfall_fraction: float,
    shuttle_delay_hours: float,
    staff_shortfall_fraction: float,
    staff_delay_hours: float,
    vot_cny_per_hour: float,
) -> dict[str, float]:
    """Estimate visitor time loss and its VOT-valued currency equivalent.

    The result measures scenario traveller inconvenience only.  It must not be
    interpreted as an operator cash ledger, compensation amount, or survey-based
    willingness-to-pay estimate.
    """
    visitors = _non_negative(daily_visitors, "daily_visitors")
    unserved_spaces = _non_negative(parking_unserved_spaces, "parking_unserved_spaces")
    party_size = float(average_party_size)
    if party_size <= 0:
        raise ValueError("average_party_size must be positive")
    parking_delay = _non_negative(parking_delay_hours, "parking_delay_hours")
    transfer = _share(transfer_share, "transfer_share")
    shuttle_shortfall = _share(shuttle_shortfall_fraction, "shuttle_shortfall_fraction")
    shuttle_delay = _non_negative(shuttle_delay_hours, "shuttle_delay_hours")
    staff_shortfall = _share(staff_shortfall_fraction, "staff_shortfall_fraction")
    staff_delay = _non_negative(staff_delay_hours, "staff_delay_hours")
    vot = _non_negative(vot_cny_per_hour, "vot_cny_per_hour")

    parking_hours = unserved_spaces * party_size * parking_delay
    shuttle_hours = visitors * transfer * shuttle_shortfall * shuttle_delay
    staff_hours = visitors * staff_shortfall * staff_delay
    total_hours = parking_hours + shuttle_hours + staff_hours
    return {
        "parking_time_loss_hours": parking_hours,
        "shuttle_time_loss_hours": shuttle_hours,
        "staff_time_loss_hours": staff_hours,
        "traveler_time_loss_hours": total_hours,
        "traveler_time_loss_cny": total_hours * vot,
    }
