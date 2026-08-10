"""Auditable descriptive analysis for Question 1 tourism time-series outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("input data frame is empty")


def analyze_city_annual_trend(source: pd.DataFrame, value_column: str = "数值") -> pd.DataFrame:
    """Return official annual totals with a transparent three-year smooth trend."""
    _require_columns(source, {"year", value_column})
    result = source.loc[:, ["year", value_column]].copy()
    result["year"] = pd.to_numeric(result["year"], errors="raise").astype(int)
    result["official_tourists"] = pd.to_numeric(result[value_column], errors="raise")
    result = result.sort_values("year").drop_duplicates("year", keep="last")
    result["trend_component"] = result["official_tourists"].rolling(3, center=True, min_periods=1).mean()
    result["random_component"] = result["official_tourists"] - result["trend_component"]
    result["unit"] = "thousand_person_visits"
    result["series_scope"] = "official_citywide_annual_domestic_tourists"
    return result.loc[:, ["year", "official_tourists", "trend_component", "random_component", "unit", "series_scope"]].reset_index(drop=True)


def decompose_regional_pressure(source: pd.DataFrame, rolling_window: int = 31) -> pd.DataFrame:
    """Additively decompose log regional pressure into trend, month season and residual."""
    _require_columns(source, {"date", "region_code", "pressure_index"})
    if rolling_window <= 0:
        raise ValueError("rolling_window must be positive")
    result = source.loc[:, ["date", "region_code", "pressure_index"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["pressure_index"] = pd.to_numeric(result["pressure_index"], errors="raise").clip(lower=0)
    result = result.sort_values(["region_code", "date"]).reset_index(drop=True)
    result["log_pressure"] = np.log1p(result["pressure_index"])
    result["trend_component"] = result.groupby("region_code", group_keys=False)["log_pressure"].transform(
        lambda values: values.rolling(rolling_window, center=True, min_periods=1).mean()
    )
    result["month"] = result["date"].dt.month
    detrended = result["log_pressure"] - result["trend_component"]
    seasonal_lookup = detrended.groupby([result["region_code"], result["month"]]).mean()
    result["seasonal_component"] = [seasonal_lookup.loc[(region, month)] for region, month in zip(result["region_code"], result["month"])]
    result["random_component"] = result["log_pressure"] - result["trend_component"] - result["seasonal_component"]
    result["series_scope"] = "estimated_regional_daily_relative_pressure"
    return result.loc[:, ["date", "region_code", "pressure_index", "log_pressure", "trend_component", "seasonal_component", "random_component", "series_scope"]]


def classify_regional_seasons(source: pd.DataFrame) -> pd.DataFrame:
    """Classify each region-month from its multi-year median relative pressure."""
    _require_columns(source, {"date", "region_code", "pressure_index"})
    result = source.loc[:, ["date", "region_code", "pressure_index"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result["pressure_index"] = pd.to_numeric(result["pressure_index"], errors="raise").clip(lower=0)
    result["month"] = result["date"].dt.month
    monthly = result.groupby(["region_code", "month"], as_index=False)["pressure_index"].median().rename(columns={"pressure_index": "monthly_median_pressure"})
    quantiles = monthly.groupby("region_code")["monthly_median_pressure"].quantile([0.25, 0.75]).unstack().rename(columns={0.25: "q25", 0.75: "q75"})
    monthly = monthly.join(quantiles, on="region_code")
    monthly["season_label"] = np.select(
        [
            monthly["monthly_median_pressure"] > monthly["q75"],
            (monthly["monthly_median_pressure"] < monthly["q25"]) |
            (monthly["monthly_median_pressure"].eq(0) & monthly["monthly_median_pressure"].eq(monthly["q25"])),
        ],
        ["peak_season", "off_season"], default="shoulder_season",
    )
    monthly["classification_basis"] = "region_specific_2023_2025_monthly_median_pressure_quantiles"
    return monthly.loc[:, ["region_code", "month", "monthly_median_pressure", "q25", "q75", "season_label", "classification_basis"]].sort_values(["region_code", "month"]).reset_index(drop=True)


def rank_factor_strength(parameters: pd.DataFrame) -> pd.DataFrame:
    """Rank fitted covariates by absolute standardized joint-model coefficient."""
    _require_columns(parameters, {"parameter_group", "parameter", "estimate_standardized"})
    result = parameters.loc[
        parameters["parameter_group"].eq("covariate") & ~parameters["parameter"].eq("intercept"),
        ["parameter", "estimate_standardized"],
    ].copy()
    result["estimate_standardized"] = pd.to_numeric(result["estimate_standardized"], errors="raise")
    result["absolute_standardized_effect"] = result["estimate_standardized"].abs()
    result["effect_direction"] = np.where(result["estimate_standardized"] >= 0, "positive", "negative")
    result["rank"] = result["absolute_standardized_effect"].rank(method="first", ascending=False).astype(int)
    result["interpretation"] = "joint_model_standardized_association_not_causal_effect"
    return result.sort_values("rank").reset_index(drop=True)


def build_service_time_scenarios(evidence: pd.DataFrame) -> pd.DataFrame:
    """Preserve service-time evidence as scenarios without inventing hourly counts."""
    required = {"region_name", "period_start", "period_end", "day_type", "time_start", "time_end", "metric_name", "evidence_type", "data_status", "source_url", "is_hourly_count"}
    _require_columns(evidence, required)
    result = evidence.loc[:, sorted(required)].copy()
    result["period_start"] = pd.to_datetime(result["period_start"], errors="raise")
    result["period_end"] = pd.to_datetime(result["period_end"], errors="raise")
    result["hourly_visitor_count"] = np.nan
    result["not_observed_hourly_flow"] = True
    result["use_in_q1"] = "service_window_or_peak_time_scenario_only"
    return result.sort_values(["region_name", "period_start", "time_start"]).reset_index(drop=True)


def construct_annual_constrained_city_daily_series(
    regional_estimates: pd.DataFrame, annual_totals: pd.DataFrame,
    annual_value_column: str = "数值",
) -> pd.DataFrame:
    """Scale the three-core-area daily shape to each available official annual total."""
    _require_columns(regional_estimates, {"date", "region_code", "visitor_estimate_baseline"})
    _require_columns(annual_totals, {"year", annual_value_column})
    regional = regional_estimates.loc[:, ["date", "region_code", "visitor_estimate_baseline"]].copy()
    regional["date"] = pd.to_datetime(regional["date"], errors="raise")
    regional["visitor_estimate_baseline"] = pd.to_numeric(regional["visitor_estimate_baseline"], errors="raise").clip(lower=0)
    daily = regional.groupby("date", as_index=False)["visitor_estimate_baseline"].sum().rename(columns={"visitor_estimate_baseline": "three_region_daily_shape"})
    daily["year"] = daily["date"].dt.year
    official = annual_totals.loc[:, ["year", annual_value_column]].copy()
    official["year"] = pd.to_numeric(official["year"], errors="raise").astype(int)
    official["official_annual_tourists_thousand"] = pd.to_numeric(official[annual_value_column], errors="raise")
    daily = daily.merge(official.loc[:, ["year", "official_annual_tourists_thousand"]], on="year", how="inner")
    annual_shape = daily.groupby("year")["three_region_daily_shape"].transform("sum")
    if annual_shape.le(0).any():
        raise ValueError("regional annual shape must be positive for every constrained year")
    daily["annual_scale_factor"] = daily["official_annual_tourists_thousand"] * 1000 / annual_shape
    daily["city_daily_visitor_estimate"] = daily["three_region_daily_shape"] * daily["annual_scale_factor"]
    daily["estimate_label"] = "annual_anchor_constrained_city_daily_estimate"
    daily["series_scope"] = "citywide_daily_estimate__official_annual_total_constrained__three_core_regions_supply_shape"
    return daily.loc[:, ["date", "year", "three_region_daily_shape", "official_annual_tourists_thousand", "annual_scale_factor", "city_daily_visitor_estimate", "estimate_label", "series_scope"]].sort_values("date").reset_index(drop=True)


def _clock_minutes(value: object) -> int:
    hour, minute = str(value).split(":")[:2]
    return int(hour) * 60 + int(minute)


def _clock_text(minutes: int) -> str:
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _triangular_weight(position: np.ndarray, left: float, mode: float, right: float) -> np.ndarray:
    rising = (position - left) / (mode - left)
    falling = (right - position) / (right - mode)
    return np.maximum(0.0, np.minimum(rising, falling))


def build_entry_exit_time_profiles(evidence: pd.DataFrame, bin_minutes: int = 30) -> pd.DataFrame:
    """Create normalized, assumption-driven intra-day entry/exit service profiles."""
    if bin_minutes <= 0:
        raise ValueError("bin_minutes must be positive")
    scenarios = build_service_time_scenarios(evidence)
    profiles: list[dict[str, object]] = []
    for scenario_id, row in scenarios.reset_index(drop=True).iterrows():
        start, end = _clock_minutes(row["time_start"]), _clock_minutes(row["time_end"])
        if end <= start:
            raise ValueError("time_end must be after time_start")
        metric = str(row["metric_name"])
        if "日出" in metric:
            scenario_type, entry_shape, exit_shape = "sunrise_viewing", (0.0, 0.18, 0.55), (0.30, 0.65, 1.0)
        elif "夜" in metric or "演艺" in metric:
            scenario_type, entry_shape, exit_shape = "night_tourism", (0.0, 0.50, 0.85), (0.30, 0.88, 1.0)
        else:
            scenario_type, entry_shape, exit_shape = "daytime_service", (0.0, 0.25, 0.70), (0.30, 0.78, 1.0)
        starts = np.arange(start, end, bin_minutes)
        ends = np.minimum(starts + bin_minutes, end)
        midpoint = (starts + ends) / 2
        position = (midpoint - start) / (end - start)
        entry = _triangular_weight(position, *entry_shape) * (ends - starts)
        exit_ = _triangular_weight(position, *exit_shape) * (ends - starts)
        entry = entry / entry.sum(); exit_ = exit_ / exit_.sum()
        for index in range(len(starts)):
            profiles.append({
                "scenario_id": scenario_id,
                "region_name": row["region_name"],
                "period_start": row["period_start"], "period_end": row["period_end"],
                "day_type": row["day_type"], "scenario_type": scenario_type,
                "time_bin_start": _clock_text(int(starts[index])), "time_bin_end": _clock_text(int(ends[index])),
                "entry_weight": float(entry[index]), "exit_weight": float(exit_[index]),
                "entry_peak_fraction": entry_shape[1], "exit_peak_fraction": exit_shape[1],
                "entry_peak_fraction_sensitivity": "base_plus_or_minus_0.10_of_service_window",
                "profile_status": "assumption_driven_not_observed_hourly_flow",
                "evidence_type": row["evidence_type"], "data_status": row["data_status"], "source_url": row["source_url"],
            })
    return pd.DataFrame(profiles)


def build_q1_analysis_outputs(
    annual: pd.DataFrame, pressure: pd.DataFrame, parameters: pd.DataFrame,
    service_evidence: pd.DataFrame, output_directory: str | Path,
    annual_value_column: str = "数值", regional_visitor_estimates: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Build the complete auditable table set for Question 1 descriptive analysis."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    city = analyze_city_annual_trend(annual, value_column=annual_value_column)
    decomposition = decompose_regional_pressure(pressure)
    seasons = classify_regional_seasons(pressure)
    factors = rank_factor_strength(parameters)
    service = build_service_time_scenarios(service_evidence)
    regional_visitor_estimates = pressure if regional_visitor_estimates is None else regional_visitor_estimates
    city_daily = construct_annual_constrained_city_daily_series(regional_visitor_estimates, annual, annual_value_column)
    city_for_decomposition = city_daily.loc[:, ["date", "city_daily_visitor_estimate"]].rename(columns={"city_daily_visitor_estimate": "pressure_index"})
    city_for_decomposition["region_code"] = "CITY"
    city_decomposition = decompose_regional_pressure(city_for_decomposition).drop(columns="region_code")
    city_monthly = city_daily.assign(month=city_daily["date"].dt.month).groupby(["year", "month"], as_index=False)["city_daily_visitor_estimate"].sum()
    city_monthly["annual_city_total"] = city_monthly.groupby("year")["city_daily_visitor_estimate"].transform("sum")
    city_monthly["monthly_share"] = city_monthly["city_daily_visitor_estimate"] / city_monthly["annual_city_total"]
    entry_exit = build_entry_exit_time_profiles(service_evidence)
    city.to_csv(output / "q1_city_annual_trend.csv", index=False, encoding="utf-8-sig")
    decomposition.to_csv(output / "q1_regional_pressure_decomposition.csv", index=False, encoding="utf-8-sig")
    seasons.to_csv(output / "q1_regional_season_classification.csv", index=False, encoding="utf-8-sig")
    factors.to_csv(output / "q1_factor_strength.csv", index=False, encoding="utf-8-sig")
    service.to_csv(output / "q1_service_time_scenarios.csv", index=False, encoding="utf-8-sig")
    city_daily.to_csv(output / "q1_city_daily_annual_constrained.csv", index=False, encoding="utf-8-sig")
    city_decomposition.to_csv(output / "q1_city_daily_decomposition.csv", index=False, encoding="utf-8-sig")
    city_monthly.to_csv(output / "q1_city_monthly_cyclic_profile.csv", index=False, encoding="utf-8-sig")
    entry_exit.to_csv(output / "q1_city_entry_exit_time_profiles.csv", index=False, encoding="utf-8-sig")
    report: dict[str, object] = {
        "city_annual_rows": int(len(city)),
        "city_annual_range": [int(city["year"].min()), int(city["year"].max())],
        "regional_pressure_rows": int(len(decomposition)),
        "regional_pressure_range": [str(decomposition["date"].min().date()), str(decomposition["date"].max().date())],
        "regions": sorted(decomposition["region_code"].unique().tolist()),
        "pressure_unit": "estimated_relative_pressure_index_not_official_daily_visitors",
        "hourly_flow_claim": "not_observed",
        "city_daily_rows": int(len(city_daily)),
        "city_daily_years": sorted(city_daily["year"].unique().tolist()),
        "city_daily_method": "official_annual_total_constrained_by_three_core_region_daily_shape",
        "entry_exit_profile_method": "normalized_triangular_service_scenario_with_peak_shift_sensitivity",
        "decomposition_method": "log1p_pressure__31day_centered_moving_average__calendar_month_mean",
    }
    (output / "q1_analysis_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
