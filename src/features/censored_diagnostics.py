"""Diagnostics for sampled ranking observations with left-censored attraction rows."""

from __future__ import annotations

from math import erf, pi, sqrt

import numpy as np
import pandas as pd


def _as_bool(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)
    return values.astype("string").str.lower().isin(["true", "1", "yes"])


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def build_censored_observation_diagnostics(
    panel: pd.DataFrame,
    min_log_std: float = 0.25,
) -> pd.DataFrame:
    """Summarize observed ranking density and left-censored likelihood by sampled region-day."""
    required = {"date", "region_code", "is_observed", "visitor_index"}
    if missing := required.difference(panel.columns):
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if min_log_std <= 0:
        raise ValueError("min_log_std must be positive")

    source = panel.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["visitor_index"] = pd.to_numeric(source["visitor_index"], errors="coerce")
    source["is_observed"] = _as_bool(source["is_observed"])
    rows: list[dict[str, object]] = []
    for (date, region_code), group in source.groupby(["date", "region_code"], dropna=False):
        observed_values = group.loc[group["is_observed"], "visitor_index"]
        positive_observed = observed_values.loc[observed_values.gt(0)]
        density_count = int(group["is_observed"].sum())
        censored_count = int((~group["is_observed"]).sum())
        total_count = int(len(group))
        row: dict[str, object] = {
            "date": date,
            "region_code": region_code,
            "density_count": density_count,
            "left_censored_count": censored_count,
            "censor_rate": censored_count / total_count if total_count else np.nan,
            "censor_threshold_index": np.nan,
            "log_index_mean": np.nan,
            "log_index_std": np.nan,
            "density_log_likelihood": np.nan,
            "censor_log_likelihood": np.nan,
            "total_log_likelihood": np.nan,
            "uncertainty_flag": "no_density_observation",
        }
        if density_count == 0:
            rows.append(row)
            continue
        if positive_observed.empty:
            row["uncertainty_flag"] = "no_positive_density_index"
            rows.append(row)
            continue
        log_values = np.log1p(observed_values.loc[observed_values.ge(0)].dropna().to_numpy(dtype=float))
        log_mean = float(np.mean(log_values))
        raw_std = float(np.std(log_values, ddof=1)) if density_count >= 2 else 0.0
        log_std = max(raw_std, min_log_std)
        threshold = float(positive_observed.min())
        log_threshold = float(np.log1p(threshold))
        standardized_threshold = (log_threshold - log_mean) / log_std
        cdf_probability = max(_normal_cdf(standardized_threshold), 1e-12)
        density_ll = float(
            np.sum(-0.5 * ((log_values - log_mean) / log_std) ** 2 - np.log(log_std * sqrt(2 * pi)))
        )
        censor_ll = float(censored_count * np.log(cdf_probability))
        row.update(
            {
                "censor_threshold_index": threshold,
                "log_index_mean": log_mean,
                "log_index_std": log_std,
                "density_log_likelihood": density_ll,
                "censor_log_likelihood": censor_ll,
                "total_log_likelihood": density_ll + censor_ll,
                "uncertainty_flag": "small_observed_sample" if density_count <= 2 else "estimated_log_std",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["date", "region_code"]).reset_index(drop=True)
