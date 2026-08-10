"""Forecasting primitives for Question 2 with explicit scale and provenance boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnnualLogTrend:
    """Weighted log-linear annual visitor model fitted to official totals only."""

    origin_year: int
    intercept: float
    slope: float
    residual_std: float
    x_mean: float
    weighted_sxx: float
    effective_n: float


@dataclass(frozen=True)
class DailyRidgeForecaster:
    """Regularized daily forecast model with serializable feature scaling."""

    feature_names: tuple[str, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    residual_std: float
    region_levels: tuple[str, ...]
    dynamic: bool
    prediction_floor: float


def annual_rolling_origin_splits(years: Iterable[int], min_train_years: int = 8) -> list[tuple[int, int, int]]:
    """Return chronological (first_train_year, last_train_year, target_year) splits."""
    unique_years = sorted({int(year) for year in years})
    if min_train_years < 2:
        raise ValueError("min_train_years must be at least 2")
    return [
        (unique_years[0], unique_years[target_index - 1], unique_years[target_index])
        for target_index in range(min_train_years, len(unique_years))
    ]


def fit_annual_log_trend(
    annual: pd.DataFrame,
    value_column: str,
    recency_half_life_years: float = 8.0,
) -> AnnualLogTrend:
    """Fit a recency-weighted trend to positive official annual totals."""
    if {"year", value_column}.difference(annual.columns):
        raise ValueError("annual frame must contain year and the requested value column")
    if recency_half_life_years <= 0:
        raise ValueError("recency_half_life_years must be positive")
    frame = annual.loc[:, ["year", value_column]].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame["value"] = pd.to_numeric(frame[value_column], errors="raise")
    frame = frame.loc[frame["value"] > 0].drop_duplicates("year", keep="last").sort_values("year")
    if len(frame) < 3:
        raise ValueError("at least three positive annual totals are required")
    origin_year = int(frame["year"].max())
    x = (frame["year"].to_numpy(dtype=float) - origin_year)
    y = np.log(frame["value"].to_numpy(dtype=float))
    weights = np.exp(np.log(0.5) * (origin_year - frame["year"].to_numpy(dtype=float)) / recency_half_life_years)
    x_mean = float(np.average(x, weights=weights))
    y_mean = float(np.average(y, weights=weights))
    centered_x = x - x_mean
    weighted_sxx = float(np.sum(weights * centered_x**2))
    if weighted_sxx <= 0:
        raise ValueError("annual years must not all be identical")
    slope = float(np.sum(weights * centered_x * (y - y_mean)) / weighted_sxx)
    intercept = float(y_mean - slope * x_mean)
    residual_std = float(np.sqrt(np.average((y - (intercept + slope * x)) ** 2, weights=weights)))
    effective_n = float(weights.sum() ** 2 / np.square(weights).sum())
    return AnnualLogTrend(origin_year, intercept, slope, residual_std, x_mean, weighted_sxx, effective_n)


def forecast_annual_total(model: AnnualLogTrend, target_year: int, interval_z: float = 1.645) -> dict[str, float]:
    """Forecast one official-scale annual total with an 90% log-scale interval by default."""
    if interval_z < 0:
        raise ValueError("interval_z must be non-negative")
    x_target = float(int(target_year) - model.origin_year)
    log_point = model.intercept + model.slope * x_target
    leverage = 1.0 + 1.0 / model.effective_n + (x_target - model.x_mean) ** 2 / model.weighted_sxx
    log_half_width = interval_z * model.residual_std * np.sqrt(leverage)
    return {
        "year": int(target_year),
        "point": float(np.exp(log_point)),
        "lower": float(np.exp(log_point - log_half_width)),
        "upper": float(np.exp(log_point + log_half_width)),
    }


def _annual_candidate_point(train: pd.DataFrame, value_column: str, target_year: int, candidate: str) -> float:
    ordered = train.loc[:, ["year", value_column]].copy().sort_values("year")
    values = pd.to_numeric(ordered[value_column], errors="raise").to_numpy(dtype=float)
    if candidate == "weighted_log_trend":
        return forecast_annual_total(fit_annual_log_trend(ordered, value_column), target_year)["point"]
    if candidate == "last_year_naive":
        return float(values[-1])
    if candidate == "recent_median_growth":
        growth = values[1:] / values[:-1]
        return float(values[-1] * np.median(growth[-min(3, len(growth)):]))
    raise ValueError(f"unknown annual candidate: {candidate}")


def evaluate_annual_candidates(annual: pd.DataFrame, value_column: str, min_train_years: int = 8) -> pd.DataFrame:
    """Score interpretable annual candidates by strictly forward rolling forecasts."""
    frame = annual.loc[:, ["year", value_column]].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    frame = frame.loc[frame[value_column] > 0].drop_duplicates("year", keep="last").sort_values("year")
    effective_min_train_years = min(min_train_years, len(frame) - 1)
    if effective_min_train_years < 2:
        raise ValueError("at least three positive annual totals are required for rolling validation")
    candidates = ("weighted_log_trend", "last_year_naive", "recent_median_growth")
    records: list[dict[str, object]] = []
    for candidate in candidates:
        actuals: list[float] = []; predictions: list[float] = []
        for _, train_end, target_year in annual_rolling_origin_splits(frame["year"], effective_min_train_years):
            train = frame.loc[frame["year"] <= train_end]
            actual = float(frame.loc[frame["year"].eq(target_year), value_column].iloc[0])
            predictions.append(_annual_candidate_point(train, value_column, target_year, candidate)); actuals.append(actual)
        residual = np.asarray(predictions) - np.asarray(actuals)
        scale = np.maximum((np.abs(np.asarray(predictions)) + np.abs(np.asarray(actuals))) / 2.0, 1e-9)
        records.append({"candidate": candidate, "test_rows": len(actuals), "mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual**2))), "smape": float(100.0 * np.mean(np.abs(residual) / scale))})
    return pd.DataFrame(records)


def select_annual_candidate(scores: pd.DataFrame) -> str:
    """Choose lowest sMAPE; use model simplicity then MAE/RMSE as deterministic tie-breakers."""
    required = {"candidate", "mae", "rmse", "smape"}
    if required.difference(scores.columns) or scores.empty:
        raise ValueError("annual score table is empty or incomplete")
    complexity = {"last_year_naive": 0, "recent_median_growth": 1, "weighted_log_trend": 2}
    ordered = scores.copy(); ordered["complexity"] = ordered["candidate"].map(complexity).fillna(99)
    return str(ordered.sort_values(["smape", "complexity", "mae", "rmse"]).iloc[0]["candidate"])


def forecast_annual_candidate(
    annual: pd.DataFrame, value_column: str, target_year: int, candidate: str, min_train_years: int = 8, interval_z: float = 1.645,
) -> dict[str, float | int | str]:
    """Forecast a selected annual candidate with interval width from its own rolling log residuals."""
    frame = annual.loc[:, ["year", value_column]].copy().sort_values("year")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise").astype(int)
    frame[value_column] = pd.to_numeric(frame[value_column], errors="raise")
    frame = frame.loc[frame[value_column] > 0].drop_duplicates("year", keep="last")
    point = _annual_candidate_point(frame, value_column, target_year, candidate)
    log_residuals: list[float] = []
    for _, train_end, test_year in annual_rolling_origin_splits(frame["year"], min_train_years):
        train = frame.loc[frame["year"] <= train_end]
        actual = float(frame.loc[frame["year"].eq(test_year), value_column].iloc[0])
        predicted = _annual_candidate_point(train, value_column, test_year, candidate)
        log_residuals.append(float(np.log(actual) - np.log(max(predicted, 1e-9))))
    residual_std = float(np.std(log_residuals, ddof=0)) if log_residuals else 0.0
    half_width = interval_z * residual_std
    return {"year": int(target_year), "candidate": candidate, "point": float(point), "lower": float(point * np.exp(-half_width)), "upper": float(point * np.exp(half_width)), "rolling_log_residual_std": residual_std}


def allocate_monthly_total(annual_total: float, monthly_shape: Iterable[float], year: int) -> pd.DataFrame:
    """Allocate an annual estimate by non-negative monthly weights while preserving its total."""
    total = float(annual_total)
    weights = np.asarray(list(monthly_shape), dtype=float)
    if total < 0:
        raise ValueError("annual_total must be non-negative")
    if len(weights) != 12:
        raise ValueError("monthly_shape must contain exactly 12 weights")
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("monthly_shape must be finite, non-negative, and contain positive mass")
    allocation = total * weights / weights.sum()
    allocation[-1] = total - allocation[:-1].sum()
    return pd.DataFrame({
        "year": int(year),
        "month": np.arange(1, 13, dtype=int),
        "monthly_shape_weight": weights,
        "visitor_estimate": allocation,
        "estimate_label": "official_annual_total_constrained_monthly_visitor_estimate",
    })


def build_daily_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create calendar covariates that remain known for any future forecast date."""
    if "date" not in frame.columns:
        raise ValueError("daily frame missing date")
    result = frame.copy()
    date = pd.to_datetime(result["date"], errors="raise")
    day_of_year = date.dt.dayofyear.to_numpy(dtype=float)
    result["is_summer"] = date.dt.month.isin([7, 8]).astype(float)
    result["is_weekend"] = (date.dt.dayofweek >= 5).astype(float)
    for weekday in range(1, 7):
        result[f"weekday_{weekday}"] = (date.dt.dayofweek == weekday).astype(float)
    result["annual_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.25)
    result["annual_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.25)
    return result


def _daily_design(frame: pd.DataFrame, region_levels: tuple[str, ...], dynamic: bool) -> tuple[np.ndarray, tuple[str, ...]]:
    calendar = build_daily_calendar_features(frame)
    pieces = [np.ones((len(calendar), 1), dtype=float)]
    names = ["intercept"]
    for name in ["is_summer", "annual_sin", "annual_cos", *[f"weekday_{weekday}" for weekday in range(1, 7)]]:
        pieces.append(calendar[[name]].to_numpy(dtype=float))
        names.append(name)
    if dynamic:
        for name in ["rain_mm", "temperature_c", "search_lag_1"]:
            if name in calendar.columns:
                values = pd.to_numeric(calendar[name], errors="coerce").to_numpy(dtype=float)
                finite_median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
                pieces.append(np.nan_to_num(values, nan=finite_median).reshape(-1, 1))
                names.append(name)
    if dynamic and "target_lag_1" in calendar.columns:
        values = pd.to_numeric(calendar["target_lag_1"], errors="coerce").to_numpy(dtype=float)
        finite_median = float(np.nanmedian(values)) if np.isfinite(values).any() else 0.0
        pieces.append(np.nan_to_num(values, nan=finite_median).reshape(-1, 1))
        names.append("target_lag_1")
    if "region_code" in calendar.columns:
        regions = calendar["region_code"].astype(str)
        for region in region_levels[1:]:
            pieces.append(regions.eq(region).to_numpy(dtype=float).reshape(-1, 1))
            names.append(f"region__{region}")
    return np.hstack(pieces), tuple(names)


def fit_daily_ridge_forecaster(
    frame: pd.DataFrame, target_column: str, dynamic: bool, ridge_alpha: float = 10.0,
) -> DailyRidgeForecaster:
    """Fit a log-scale daily model; dynamic=True permits lag/search/weather covariates."""
    if target_column not in frame.columns:
        raise ValueError("daily frame missing target column")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")
    target = pd.to_numeric(frame[target_column], errors="coerce")
    valid = target.notna() & target.ge(0)
    if valid.sum() < 3:
        raise ValueError("at least three non-negative daily targets are required")
    regions = tuple(sorted(frame.get("region_code", pd.Series(["CITY"] * len(frame))).astype(str).unique()))
    design, names = _daily_design(frame, regions, dynamic)
    x = design[valid.to_numpy()]
    means = x.mean(axis=0); scales = x.std(axis=0)
    means[0] = 0.0; scales[0] = 1.0
    scales[scales == 0] = 1.0
    standardized = (x - means) / scales
    standardized[:, 0] = 1.0
    y = np.log1p(target.loc[valid].to_numpy(dtype=float))
    penalty = np.eye(standardized.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(standardized.T @ standardized + penalty) @ standardized.T @ y
    residual_std = float(np.std(y - standardized @ coefficients, ddof=0))
    positive_target = target.loc[valid & target.gt(0)].to_numpy(dtype=float)
    floor = float(np.quantile(positive_target, 0.05)) if len(positive_target) else 0.0
    return DailyRidgeForecaster(names, tuple(means), tuple(scales), tuple(coefficients), residual_std, regions, dynamic, floor)


def predict_daily_ridge_forecaster(model: DailyRidgeForecaster, frame: pd.DataFrame, interval_z: float = 1.645) -> pd.DataFrame:
    """Return direct multi-step daily forecasts using only supplied scenario covariates."""
    if interval_z < 0:
        raise ValueError("interval_z must be non-negative")
    design, names = _daily_design(frame, model.region_levels, model.dynamic)
    if names != model.feature_names:
        raise ValueError("prediction feature columns differ from fitted feature columns")
    means = np.asarray(model.feature_means); scales = np.asarray(model.feature_scales)
    standardized = (design - means) / scales
    standardized[:, 0] = 1.0
    log_point = standardized @ np.asarray(model.coefficients)
    half_width = interval_z * model.residual_std
    result = frame.loc[:, [name for name in ["date", "region_code"] if name in frame.columns]].copy()
    result["prediction_q10"] = np.maximum(np.expm1(log_point - half_width), 0.0)
    result["prediction_q50"] = np.maximum(np.expm1(log_point), model.prediction_floor)
    result["prediction_q90"] = np.maximum(np.expm1(log_point + half_width), result["prediction_q50"])
    result["estimate_label"] = "anchor_constrained_daily_visitor_forecast"
    result["forecast_method"] = "dynamic_ridge" if model.dynamic else "seasonal_calendar_ridge"
    return result


def scenario_adjust_weather(frame: pd.DataFrame, rain_multiplier: float = 1.0, rain_add_mm: float = 0.0, temperature_shift_c: float = 0.0) -> pd.DataFrame:
    """Create a weather-only scenario without changing date, search, or regional inputs."""
    if rain_multiplier < 0 or rain_add_mm < 0:
        raise ValueError("rain_multiplier and rain_add_mm must be non-negative")
    result = frame.copy()
    if "rain_mm" in result.columns:
        result["rain_mm"] = pd.to_numeric(result["rain_mm"], errors="coerce") * rain_multiplier + rain_add_mm
    if "temperature_c" in result.columns:
        result["temperature_c"] = pd.to_numeric(result["temperature_c"], errors="coerce") + temperature_shift_c
    return result
