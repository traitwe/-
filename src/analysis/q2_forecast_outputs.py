"""Auditable two-scale forecast outputs for Question 2."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from src.models.question2_forecasting import (
    allocate_monthly_total,
    evaluate_annual_candidates,
    fit_daily_ridge_forecaster,
    forecast_annual_candidate,
    predict_daily_ridge_forecaster,
    scenario_adjust_weather,
    select_annual_candidate,
)


def _score(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    residual = prediction.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    denominator = np.maximum((np.abs(actual.to_numpy(dtype=float)) + np.abs(prediction.to_numpy(dtype=float))) / 2.0, 1e-9)
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual**2))), "smape": float(100.0 * np.mean(np.abs(residual) / denominator))}


def _monthly_shape(regional: pd.DataFrame, end_year: int) -> np.ndarray:
    frame = regional.copy(); frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[frame["date"].dt.year <= end_year].copy()
    frame["month"] = frame["date"].dt.month
    weights = frame.groupby("month")["pressure_index"].sum().reindex(range(1, 13), fill_value=0.0).to_numpy(dtype=float)
    return weights if weights.sum() > 0 else np.ones(12, dtype=float)


def build_climatological_covariates(covariates: pd.DataFrame, forecast_year: int) -> pd.DataFrame:
    """Build future day-level weather/search scenarios from pre-forecast calendar-day medians only."""
    if "date" not in covariates.columns:
        raise ValueError("covariates missing date")
    source = covariates.copy(); source["date"] = pd.to_datetime(source["date"], errors="raise")
    source = source.loc[source["date"].dt.year < forecast_year].copy()
    feature_columns = [column for column in ["rain_mm", "temperature_c", "search_lag_1"] if column in source.columns]
    dates = pd.DataFrame({"date": pd.date_range(f"{forecast_year}-01-01", f"{forecast_year}-12-31", freq="D")})
    if not feature_columns:
        return dates
    source["month"] = source["date"].dt.month; source["day"] = source["date"].dt.day
    profile = source.groupby(["month", "day"], as_index=False)[feature_columns].median()
    dates["month"] = dates["date"].dt.month; dates["day"] = dates["date"].dt.day
    result = dates.merge(profile, on=["month", "day"], how="left")
    monthly = source.groupby(source["date"].dt.month)[feature_columns].median()
    for column in feature_columns:
        result[column] = result[column].fillna(result["month"].map(monthly[column])).fillna(float(source[column].median()))
    return result.drop(columns=["month", "day"])


def build_external_anchor_validation(anchors: pd.DataFrame, city_daily: pd.DataFrame, regional_daily: pd.DataFrame) -> pd.DataFrame:
    """Compare reported anchors only where their geographical and metric scopes truly match."""
    required = {"period_start", "period_end", "region_name", "visitor_count_10k_persons", "value_qualifier", "metric_scope", "source_url"}
    if required.difference(anchors.columns):
        raise ValueError("anchor table is incomplete")
    city = city_daily.copy(); city["date"] = pd.to_datetime(city["date"], errors="raise")
    regional = regional_daily.copy(); regional["date"] = pd.to_datetime(regional["date"], errors="raise")
    region_map = {"Beidaihe_District": "BDH", "Shanhaiguan_District": "SHG", "Beidaihe_New_District": "HGA", "Haigang_Aranya": "HGA"}
    rows: list[dict[str, object]] = []
    for _, anchor in anchors.iterrows():
        start, end = pd.Timestamp(anchor["period_start"]), pd.Timestamp(anchor["period_end"])
        reported = float(anchor["visitor_count_10k_persons"]) * 10000.0
        name, scope = str(anchor["region_name"]), str(anchor["metric_scope"])
        direct = name == "Qinhuangdao_City" and scope == "citywide_tourist_visitors" and str(anchor["value_qualifier"]) == "exact"
        if name == "Qinhuangdao_City":
            values = city.loc[city["date"].between(start, end), "city_daily_visitor_estimate"]
        else:
            code = region_map.get(name)
            values = regional.loc[regional["date"].between(start, end) & regional["region_code"].eq(code), "visitor_estimate_baseline"] if code else pd.Series(dtype=float)
        estimate = float(values.sum()) if not values.empty else np.nan
        if direct and np.isfinite(estimate):
            status, error = "directly_comparable", estimate / reported - 1.0
        elif not np.isfinite(estimate):
            status, error = "model_series_not_available_for_period", np.nan
        else:
            status, error = "scope_not_directly_comparable", np.nan
        rows.append({"period_start": start, "period_end": end, "region_name": name, "metric_scope": scope, "reported_visitor_count": reported, "model_period_estimate": estimate, "comparison_status": status, "relative_error": error, "source_url": anchor["source_url"]})
    return pd.DataFrame(rows)


def _attach_covariates(frame: pd.DataFrame, covariates: pd.DataFrame | None) -> pd.DataFrame:
    if covariates is None:
        return frame.copy()
    result = frame.copy(); result["date"] = pd.to_datetime(result["date"], errors="raise")
    source = covariates.copy(); source["date"] = pd.to_datetime(source["date"], errors="raise")
    columns = [column for column in ["date", "rain_mm", "temperature_c", "search_lag_1"] if column in source.columns]
    return result.merge(source.loc[:, columns].drop_duplicates("date"), on="date", how="left")


def _daily_comparison(regional: pd.DataFrame, observed: pd.DataFrame, covariates: pd.DataFrame | None = None) -> tuple[pd.DataFrame, bool]:
    observed_frame = observed.loc[:, ["date", "region_code", "visitor_index"]].copy()
    observed_frame["date"] = pd.to_datetime(observed_frame["date"], errors="raise")
    observed_frame["visitor_index"] = pd.to_numeric(observed_frame["visitor_index"], errors="coerce")
    observed_frame = observed_frame.dropna(subset=["visitor_index"])
    regional_dates = pd.to_datetime(regional["date"], errors="raise")
    observed_frame = observed_frame.loc[
        observed_frame["region_code"].astype(str).isin(regional["region_code"].astype(str).unique())
        & observed_frame["date"].between(regional_dates.min(), regional_dates.max())
    ].copy()
    if observed_frame.empty:
        return pd.DataFrame(columns=["candidate", "test_rows", "mae", "rmse", "smape"]), True
    holdout_year = int(observed_frame["date"].dt.year.max())
    first_test_date = pd.Timestamp(f"{holdout_year}-01-01")
    observed_frame = observed_frame.loc[observed_frame["date"].dt.year.eq(holdout_year)].copy()
    train = regional.loc[pd.to_datetime(regional["date"], errors="raise") < first_test_date].copy()
    train = _attach_covariates(train, covariates)
    if len(train) < 3:
        return pd.DataFrame(columns=["candidate", "test_rows", "mae", "rmse", "smape"]), True
    features = _attach_covariates(observed_frame.loc[:, ["date", "region_code"]], covariates)
    rows = []
    for name, dynamic in [("seasonal_calendar_ridge", False), ("dynamic_ridge", True)]:
        model = fit_daily_ridge_forecaster(train, "pressure_index", dynamic=dynamic)
        predicted = predict_daily_ridge_forecaster(model, features)
        merged = observed_frame.merge(predicted, on=["date", "region_code"], how="inner")
        if merged.empty:
            continue
        row = {"candidate": name, "test_rows": int(len(merged)), **_score(merged["visitor_index"], merged["prediction_q50"])}
        rows.append(row)
    comparison = pd.DataFrame(rows)
    return comparison, comparison.empty


def build_q2_diagnostic_figures(monthly: pd.DataFrame, daily: pd.DataFrame, output_directory: str | Path) -> None:
    """Render two compact paper figures from forecast tables, never as observed counts."""
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.bar(monthly["month"], monthly["visitor_estimate"], color="#4C78A8")
    axis.set(xlabel="Month", ylabel="Estimated visitors", title="City monthly forecast constrained by annual total")
    if {"annual_forecast_lower", "annual_forecast_upper"}.issubset(monthly.columns):
        interval = f"Annual 90% interval: {monthly['annual_forecast_lower'].iloc[0]:,.0f}–{monthly['annual_forecast_upper'].iloc[0]:,.0f}"
        axis.text(0.02, 0.95, interval, transform=axis.transAxes, va="top")
    figure.tight_layout(); figure.savefig(output / "q2_city_monthly_forecast.png", dpi=220); plt.close(figure)
    frame = daily.copy(); frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[frame["date"].dt.month.isin([7, 8])]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    for region, group in frame.groupby("region_code"):
        group = group.sort_values("date")
        axis.plot(group["date"], group["visitor_estimate_q50"], label=str(region))
        axis.fill_between(group["date"], group["visitor_estimate_q10"], group["visitor_estimate_q90"], alpha=0.16)
    axis.set(xlabel="Date", ylabel="Anchor-constrained estimated visitors", title="Summer regional daily forecast: 10%–90% interval")
    axis.legend(ncol=3, fontsize=8); figure.autofmt_xdate(); figure.tight_layout()
    figure.savefig(output / "q2_summer_regional_daily_forecast.png", dpi=220); plt.close(figure)


def build_question2_outputs(
    annual: pd.DataFrame,
    regional: pd.DataFrame,
    observed: pd.DataFrame,
    output_directory: str | Path,
    forecast_year: int = 2026,
    annual_value_column: str = "official_total",
    daily_covariates: pd.DataFrame | None = None,
) -> dict[str, object]:
    """Create Question 2 tables with explicit forecast and validation provenance."""
    required_annual = {"year", annual_value_column}
    required_regional = {"date", "region_code", "pressure_index", "baseline_scale"}
    if required_annual.difference(annual.columns) or required_regional.difference(regional.columns):
        raise ValueError("input tables do not contain the required Question 2 columns")
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    annual_comparison = evaluate_annual_candidates(annual, annual_value_column)
    annual_comparison.to_csv(output / "q2_annual_model_comparison.csv", index=False, encoding="utf-8-sig")
    annual_selected_model = select_annual_candidate(annual_comparison)
    annual_forecast = forecast_annual_candidate(annual, annual_value_column, forecast_year, annual_selected_model)
    monthly = allocate_monthly_total(annual_forecast["point"], _monthly_shape(regional, forecast_year - 1), forecast_year)
    monthly["annual_forecast_lower"] = annual_forecast["lower"]
    monthly["annual_forecast_upper"] = annual_forecast["upper"]
    monthly.to_csv(output / f"q2_city_monthly_forecast_{forecast_year}.csv", index=False, encoding="utf-8-sig")

    comparison, no_independent_observation = _daily_comparison(regional, observed, daily_covariates)
    comparison.to_csv(output / "q2_daily_model_comparison.csv", index=False, encoding="utf-8-sig")
    selected_dynamic = bool(comparison.sort_values("mae").iloc[0]["candidate"] == "dynamic_ridge") if not comparison.empty else True
    train = regional.loc[pd.to_datetime(regional["date"], errors="raise").dt.year < forecast_year].copy()
    train = _attach_covariates(train, daily_covariates)
    daily_model = fit_daily_ridge_forecaster(train, "pressure_index", dynamic=selected_dynamic)
    regions = sorted(train["region_code"].astype(str).unique())
    future_dates = pd.date_range(f"{forecast_year}-01-01", f"{forecast_year}-12-31", freq="D")
    future = pd.DataFrame({"date": np.repeat(future_dates, len(regions)), "region_code": regions * len(future_dates)})
    future_covariates = build_climatological_covariates(daily_covariates, forecast_year) if daily_covariates is not None else None
    future = _attach_covariates(future, future_covariates)
    daily = predict_daily_ridge_forecaster(daily_model, future)
    scales = train.assign(date=pd.to_datetime(train["date"], errors="raise")).sort_values("date").groupby("region_code", as_index=False)["baseline_scale"].last()
    daily = daily.merge(scales, on="region_code", how="left")
    for quantile in ["q10", "q50", "q90"]:
        daily[f"visitor_estimate_{quantile}"] = daily[f"prediction_{quantile}"] * daily["baseline_scale"]
    daily["forecast_provenance"] = "pressure_forecast_scaled_by_latest_available_region_anchor"
    daily.to_csv(output / f"q2_region_daily_forecast_{forecast_year}.csv", index=False, encoding="utf-8-sig")

    summer = daily.loc[pd.to_datetime(daily["date"]).dt.month.isin([7, 8])].groupby("region_code", as_index=False).agg(
        summer_peak_q10=("visitor_estimate_q10", "max"), summer_peak_q50=("visitor_estimate_q50", "max"), summer_peak_q90=("visitor_estimate_q90", "max"),
    )
    summer.to_csv(output / f"q2_summer_peak_range_{forecast_year}.csv", index=False, encoding="utf-8-sig")
    weather_model = daily_model
    weather_response_model = "selected_daily_model"
    if not {"rain_mm", "temperature_c"}.intersection(daily_model.feature_names) and daily_covariates is not None:
        weather_model = fit_daily_ridge_forecaster(train, "pressure_index", dynamic=True)
        weather_response_model = "dynamic_ridge_sensitivity_only"
    weather_base = predict_daily_ridge_forecaster(weather_model, future).merge(scales, on="region_code", how="left")
    rainy = scenario_adjust_weather(future, rain_multiplier=3.0)
    rain_daily = predict_daily_ridge_forecaster(weather_model, rainy).merge(scales, on="region_code", how="left")
    scenario = daily.loc[:, ["date", "region_code", "visitor_estimate_q50"]].rename(columns={"visitor_estimate_q50": "baseline_estimate"})
    scenario["weather_response_baseline_estimate"] = weather_base["prediction_q50"].to_numpy() * weather_base["baseline_scale"].to_numpy()
    scenario["rain_shock_estimate"] = rain_daily["prediction_q50"].to_numpy() * rain_daily["baseline_scale"].to_numpy()
    heat = scenario_adjust_weather(future, temperature_shift_c=5.0)
    heat_daily = predict_daily_ridge_forecaster(weather_model, heat).merge(scales, on="region_code", how="left")
    scenario["heat_shock_estimate"] = heat_daily["prediction_q50"].to_numpy() * heat_daily["baseline_scale"].to_numpy()
    scenario["scenario_label"] = "threefold_rainfall_and_plus5c_weather_shocks"
    scenario.to_csv(output / f"q2_weather_shock_{forecast_year}.csv", index=False, encoding="utf-8-sig")
    build_q2_diagnostic_figures(monthly, daily, output)
    report = {
        "forecast_year": forecast_year,
        "annual_forecast_point": annual_forecast["point"],
        "annual_forecast_interval": [annual_forecast["lower"], annual_forecast["upper"]],
        "annual_selected_model": annual_selected_model,
        "daily_selected_model": "dynamic_ridge" if selected_dynamic else "seasonal_calendar_ridge",
        "daily_validation_scope": "raw_attraction_observation_dates_only" if not no_independent_observation else "no_independent_observation_available",
        "daily_validation_covariate_mode": "conditional_on_realized_weather_and_lagged_search" if daily_covariates is not None else "calendar_only",
        "weather_response_model": weather_response_model,
        "weather_shock_effective": bool({"rain_mm", "temperature_c"}.intersection(weather_model.feature_names)),
        "regional_daily_unit": "anchor_constrained_visitor_scale_estimate_not_observed_count",
    }
    (output / "q2_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
