"""Counterfactual visitor shocks and baseline-plan adequacy measures for Q4."""

from __future__ import annotations

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
