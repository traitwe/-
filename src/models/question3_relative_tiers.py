"""Relative service-tier rules for the reconstructed Question 3."""

from __future__ import annotations

import numpy as np
import pandas as pd


def classify_relative_pressure(frame: pd.DataFrame, value_column: str = "visitor_estimate_q50") -> pd.DataFrame:
    """Classify forecast pressure against each region's own median level."""
    if {"region_code", value_column}.difference(frame.columns):
        raise ValueError("forecast frame must contain region_code and the pressure value")
    result = frame.copy()
    result[value_column] = pd.to_numeric(result[value_column], errors="raise")
    median = result.groupby("region_code")[value_column].transform("median")
    if median.le(0).any():
        raise ValueError("regional median forecast must be positive")
    result["relative_pressure"] = result[value_column] / median
    result["pressure_tier"] = np.select(
        [result["relative_pressure"] < 0.8, result["relative_pressure"] < 1.0, result["relative_pressure"] < 1.4],
        ["low", "normal", "high"], default="extreme",
    )
    return result


def beidaihe_summer_trigger(relative_pressure: float, is_weekend: bool, is_holiday: bool) -> str:
    """Return the operational tier for Beidaihe summer service coordination."""
    if relative_pressure >= 1.8 and (is_weekend or is_holiday):
        return "emergency"
    if relative_pressure >= 1.4:
        return "surge"
    return "normal"
