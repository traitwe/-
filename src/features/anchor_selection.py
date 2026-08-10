"""Scope screening for absolute-scale visitor anchors."""

from __future__ import annotations

import pandas as pd


DIRECT_METRIC_SCOPES = {"", "district_tourist_visitors", "district_tourism_peak_season"}
DIRECT_FACILITY_SCOPES = {"", "all_visitors", "all_tourism_area"}


def classify_direct_region_anchors(
    anchors: pd.DataFrame,
    target_regions: set[str],
    model_start: str,
    model_end: str,
) -> pd.DataFrame:
    """Label whether each source anchor can directly calibrate a target regional visitor scale."""
    required = {"region_code", "period_start", "period_end", "anchor_use"}
    if missing := required.difference(anchors.columns):
        raise ValueError(f"anchors missing columns: {sorted(missing)}")
    result = anchors.copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="coerce")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="coerce")
    metric_scope = result.get("metric_scope", pd.Series("", index=result.index)).fillna("").astype(str)
    facility_scope = result.get("facility_or_scope", pd.Series("", index=result.index)).fillna("").astype(str)
    in_window = result["period_end"].ge(pd.Timestamp(model_start)) & result["period_start"].le(pd.Timestamp(model_end))
    result["scope_decision"] = "subarea_or_scenic_scope"
    result.loc[~result["region_code"].isin(target_regions), "scope_decision"] = "outside_target_regions"
    result.loc[result["anchor_use"].eq("lower_bound_only"), "scope_decision"] = "lower_bound_only"
    result.loc[~in_window, "scope_decision"] = "outside_model_window"
    direct = (
        result["region_code"].isin(target_regions)
        & result["anchor_use"].eq("point_calibration")
        & in_window
        & metric_scope.isin(DIRECT_METRIC_SCOPES)
        & facility_scope.isin(DIRECT_FACILITY_SCOPES)
    )
    result.loc[direct, "scope_decision"] = "direct_region_scale"
    return result
