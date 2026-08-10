"""Convert relative regional pressure into explicitly anchor-constrained scale scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_daily_visitor_scale(
    pressure: pd.DataFrame,
    anchors: pd.DataFrame,
    single_anchor_width: float = 0.20,
    carry_forward_width: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate per-region-year pressure scales using only direct-scope anchor periods."""
    required_pressure = {"date", "region_code", "pressure_index"}
    required_anchors = {"region_code", "period_start", "period_end", "visitor_total", "scope_decision"}
    if missing := required_pressure.difference(pressure.columns):
        raise ValueError(f"pressure missing columns: {sorted(missing)}")
    if missing := required_anchors.difference(anchors.columns):
        raise ValueError(f"anchors missing columns: {sorted(missing)}")
    if not 0 <= single_anchor_width < 1 or not 0 <= carry_forward_width < 1:
        raise ValueError("scenario widths must be in [0, 1)")

    result = pressure.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result["pressure_index"] = pd.to_numeric(result["pressure_index"], errors="coerce")
    anchor_frame = anchors.loc[anchors["scope_decision"].eq("direct_region_scale")].copy()
    anchor_frame["period_start"] = pd.to_datetime(anchor_frame["period_start"], errors="coerce")
    anchor_frame["period_end"] = pd.to_datetime(anchor_frame["period_end"], errors="coerce")
    anchor_frame["visitor_total"] = pd.to_numeric(anchor_frame["visitor_total"], errors="coerce")
    anchor_frame = anchor_frame.dropna(subset=["period_start", "period_end", "visitor_total"])

    fits: list[dict[str, object]] = []
    for anchor_id, anchor in anchor_frame.reset_index(drop=True).iterrows():
        mask = result["region_code"].eq(anchor["region_code"]) & result["date"].between(
            anchor["period_start"], anchor["period_end"]
        )
        pressure_total = float(result.loc[mask, "pressure_index"].sum())
        if pressure_total <= 0:
            continue
        fit_row = {
                "anchor_id": anchor_id + 1,
                "region_code": anchor["region_code"],
                "calibration_year": int(anchor["period_start"].year),
                "period_start": anchor["period_start"],
                "period_end": anchor["period_end"],
                "visitor_total": float(anchor["visitor_total"]),
                "pressure_period_total": pressure_total,
                "implied_scale": float(anchor["visitor_total"]) / pressure_total,
            }
        for provenance_column in ["source_dataset", "frequency", "metric_scope", "facility_or_scope"]:
            if provenance_column in anchor.index:
                fit_row[provenance_column] = anchor[provenance_column]
        fits.append(fit_row)
    anchor_fit = pd.DataFrame(fits)
    if anchor_fit.empty:
        raise ValueError("no direct anchor overlaps the supplied pressure series")

    if "frequency" not in anchor_fit.columns:
        anchor_fit["frequency"] = ""
    anchor_fit["calibration_role"] = "diagnostic_only"
    scale_rows: list[dict[str, object]] = []
    for (region_code, year), group in anchor_fit.groupby(["region_code", "calibration_year"]):
        annual_group = group.loc[group["frequency"].eq("annual")]
        primary_group = annual_group if not annual_group.empty else group
        anchor_fit.loc[primary_group.index, "calibration_role"] = "primary_scale"
        baseline_scale = float(np.exp(np.log(primary_group["implied_scale"]).mean()))
        if len(primary_group) == 1:
            low_scale = baseline_scale * (1 - single_anchor_width)
            high_scale = baseline_scale * (1 + single_anchor_width)
            interval_method = "single_anchor_scenario"
        else:
            low_scale = float(primary_group["implied_scale"].min())
            high_scale = float(primary_group["implied_scale"].max())
            interval_method = "multi_anchor_spread"
        scale_rows.append(
            {
                "region_code": region_code,
                "calibration_year": year,
                "baseline_scale": baseline_scale,
                "conservative_scale": low_scale,
                "high_scale": high_scale,
                "scale_anchor_count": int(len(primary_group)),
                "scale_interval_method": interval_method,
            }
        )
    scales = pd.DataFrame(scale_rows)
    anchor_fit = anchor_fit.merge(
        scales.loc[:, ["region_code", "calibration_year", "baseline_scale"]],
        on=["region_code", "calibration_year"],
        how="left",
    )
    anchor_fit["fitted_period_total"] = anchor_fit["pressure_period_total"] * anchor_fit["baseline_scale"]
    anchor_fit["relative_fit_error"] = (
        anchor_fit["fitted_period_total"] / anchor_fit["visitor_total"] - 1
    )

    result["calibration_year"] = result["date"].dt.year
    result = result.merge(scales, on=["region_code", "calibration_year"], how="left")
    result["scale_source_year"] = result["calibration_year"].where(result["baseline_scale"].notna())
    for region_code, indices in result.groupby("region_code").groups.items():
        row_indices = list(indices)
        subset = result.loc[row_indices].sort_values("date")
        for column in ["baseline_scale", "conservative_scale", "high_scale", "scale_anchor_count", "scale_interval_method", "scale_source_year"]:
            result.loc[subset.index, column] = subset[column].ffill()
        years_since_source = result.loc[subset.index, "calibration_year"] - result.loc[subset.index, "scale_source_year"]
        carried = years_since_source.gt(0)
        carried_indices = subset.index[carried]
        if len(carried_indices):
            result.loc[carried_indices, "conservative_scale"] *= (1 - carry_forward_width) ** years_since_source.loc[carried_indices]
            result.loc[carried_indices, "high_scale"] *= (1 + carry_forward_width) ** years_since_source.loc[carried_indices]
            result.loc[carried_indices, "scale_interval_method"] = "carry_forward_scenario"
    if result["baseline_scale"].isna().any():
        raise ValueError("some pressure rows precede the first direct anchor for their region")
    result["visitor_estimate_baseline"] = result["pressure_index"] * result["baseline_scale"]
    result["visitor_estimate_conservative"] = result["pressure_index"] * result["conservative_scale"]
    result["visitor_estimate_high"] = result["pressure_index"] * result["high_scale"]
    result["estimate_label"] = "anchor_constrained_estimate"
    return result.sort_values(["date", "region_code"]).reset_index(drop=True), anchor_fit
