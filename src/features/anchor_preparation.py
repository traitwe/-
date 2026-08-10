"""Normalize heterogeneous public visitor anchors before calibration."""

from __future__ import annotations

import pandas as pd


def prepare_calibration_anchors(frame: pd.DataFrame, region_map: dict[str, str]) -> pd.DataFrame:
    """Convert verified 10k-person values to period totals and label their calibration role."""
    required = {"period_start", "period_end", "region_name", "visitor_count_10k_persons", "frequency", "data_status"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"anchor source missing columns: {sorted(missing)}")
    result = frame.copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    result["visitor_count_10k_persons"] = pd.to_numeric(result["visitor_count_10k_persons"], errors="coerce")
    result["region_code"] = result["region_name"].map(region_map)
    result["period_days"] = (result["period_end"] - result["period_start"]).dt.days + 1
    qualifier = result.get("value_qualifier", pd.Series("", index=result.index)).astype("string")
    result["anchor_use"] = "excluded"
    valid = result["data_status"].eq("verified") & result["region_code"].notna() & result["visitor_count_10k_persons"].notna()
    result["visitor_total"] = result["visitor_count_10k_persons"] * 10_000
    daily_average = result["frequency"].eq("daily_average")
    result.loc[daily_average, "visitor_total"] = result.loc[daily_average, "visitor_total"] * result.loc[daily_average, "period_days"]
    frequency = result["frequency"].astype("string")
    lower_bound = qualifier.str.contains("greater_than", na=False) | frequency.str.contains(
        "lower_bound|cumulative_to_date", na=False
    )
    result.loc[valid & ~lower_bound, "anchor_use"] = "point_calibration"
    result.loc[valid & lower_bound, "anchor_use"] = "lower_bound_only"
    return result
