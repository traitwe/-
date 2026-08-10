"""Assemble traceable model-input tables from cleaned relative-flow and anchor data."""

from __future__ import annotations

import pandas as pd

from src.features.anchor_preparation import prepare_calibration_anchors
from src.features.ranked_observation import build_censored_observation_panel


def build_model_input_tables(
    attraction_flow: pd.DataFrame,
    visitor_anchors: pd.DataFrame,
    region_map: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Return a censored attraction panel, valid calibration anchors, and compact quality counts."""
    required_flow = {"date", "attraction_name", "region_code", "visitor_index"}
    if missing := required_flow.difference(attraction_flow.columns):
        raise ValueError(f"attraction flow missing columns: {sorted(missing)}")

    flow = attraction_flow.copy()
    flow["date"] = pd.to_datetime(flow["date"], errors="coerce")
    flow = flow.dropna(subset=["date", "attraction_name", "region_code"])
    attractions = flow.loc[:, ["attraction_name", "region_code"]].drop_duplicates()
    panel = build_censored_observation_panel(
        flow,
        attractions,
        start_date=flow["date"].min().strftime("%Y-%m-%d"),
        end_date=flow["date"].max().strftime("%Y-%m-%d"),
    )
    prepared_all = prepare_calibration_anchors(visitor_anchors, region_map)
    prepared = prepared_all.loc[
        prepared_all["anchor_use"].isin(["point_calibration", "lower_bound_only"])
    ].copy()
    quality = {
        "panel_rows": int(len(panel)),
        "observed_rows": int(panel["is_observed"].sum()),
        "censored_rows": int((~panel["is_observed"]).sum()),
        "anchor_rows": int(len(prepared)),
    }
    return panel, prepared, quality
