"""Anchor-constrained conversion from relative pressure to visitor-scale estimates."""

from __future__ import annotations

import pandas as pd


def calibrate_absolute_scale(
    pressure: pd.DataFrame,
    anchors: pd.DataFrame,
    city_annual_total: float | None = None,
    coverage_share: float | None = None,
) -> pd.DataFrame:
    """Scale each anchored region-period to its observed aggregate without claiming full-city coverage."""
    required_pressure = {"date", "region_code", "pressure_index"}
    required_anchors = {"region_code", "period_start", "period_end", "visitor_total"}
    if missing := required_pressure.difference(pressure.columns):
        raise ValueError(f"pressure missing columns: {sorted(missing)}")
    if missing := required_anchors.difference(anchors.columns):
        raise ValueError(f"anchors missing columns: {sorted(missing)}")
    if coverage_share is not None and not 0 < coverage_share <= 1:
        raise ValueError("coverage_share must be in (0, 1]")

    result = pressure.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["pressure_index"] = pd.to_numeric(result["pressure_index"], errors="coerce")
    result["visitor_estimate_baseline"] = pd.NA
    result["anchor_id"] = pd.NA

    for anchor_number, anchor in anchors.reset_index(drop=True).iterrows():
        start = pd.to_datetime(anchor["period_start"])
        end = pd.to_datetime(anchor["period_end"])
        mask = result["region_code"].eq(anchor["region_code"]) & result["date"].between(start, end)
        scale_base = result.loc[mask, "pressure_index"].sum()
        if scale_base <= 0:
            raise ValueError(f"anchor {anchor_number} has no positive pressure within its period")
        result.loc[mask, "visitor_estimate_baseline"] = (
            result.loc[mask, "pressure_index"] * float(anchor["visitor_total"]) / scale_base
        )
        result.loc[mask, "anchor_id"] = f"anchor_{anchor_number + 1}"

    result["estimate_label"] = "anchor_constrained_estimate"
    result["city_coverage_share"] = coverage_share
    result["city_annual_total_reference"] = city_annual_total
    return result
