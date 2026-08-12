"""Runnable final program for one modelling question."""
from pathlib import Path
from common_runtime import _export_submission, prepare_submission_runtime


"""Forecasting primitives for Question 2 with explicit scale and provenance boundaries."""



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
    if "region_code" in calendar.columns:
        regions = calendar["region_code"].astype(str)
        for region in region_levels[1:]:
            pieces.append(regions.eq(region).to_numpy(dtype=float).reshape(-1, 1))
            names.append(f"region__{region}")
    return np.hstack(pieces), tuple(names)


def fit_daily_ridge_forecaster(
    frame: pd.DataFrame, target_column: str, dynamic: bool, ridge_alpha: float = 10.0,
) -> DailyRidgeForecaster:
    """Fit a log-scale daily model with forecast-available dynamic covariates only."""
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


"""Auditable two-scale forecast outputs for Question 2."""



import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt












def _score(actual: pd.Series, prediction: pd.Series) -> dict[str, float]:
    residual = prediction.to_numpy(dtype=float) - actual.to_numpy(dtype=float)
    denominator = np.maximum((np.abs(actual.to_numpy(dtype=float)) + np.abs(prediction.to_numpy(dtype=float))) / 2.0, 1e-9)
    return {"mae": float(np.abs(residual).mean()), "rmse": float(np.sqrt(np.mean(residual**2))), "smape": float(100.0 * np.mean(np.abs(residual) / denominator))}


def q2_paper_figure_labels() -> dict[str, str]:
    """Return the Chinese, title-free labels used by paper figures."""
    return {
        "monthly_x": "月份",
        "monthly_y": "年度总量约束游客规模估计（人次）",
        "monthly_title": "",
        "summer_x": "日期",
        "summer_y": "锚点约束游客规模估计（人次）",
        "summer_title": "",
    }


def apply_relative_split_conformal_interval(
    prediction: pd.DataFrame,
    calibration_actual: pd.Series,
    calibration_point: pd.Series,
    coverage: float = 0.8,
) -> tuple[pd.DataFrame, float]:
    """Expand forecast intervals by a scale-free split-conformal residual radius.

    The calibration residual is expressed relative to the point prediction, so
    it can be transferred from an index-scale validation table to an
    anchor-scaled visitor forecast without mixing their units.
    """
    required = {"prediction_q10", "prediction_q50", "prediction_q90"}
    if missing := required - set(prediction.columns):
        raise ValueError(f"prediction missing columns: {sorted(missing)}")
    if not 0 < coverage < 1:
        raise ValueError("coverage must be in (0, 1)")
    actual = pd.to_numeric(calibration_actual, errors="coerce").to_numpy(dtype=float)
    point = pd.to_numeric(calibration_point, errors="coerce").to_numpy(dtype=float)
    usable = np.isfinite(actual) & np.isfinite(point) & (np.abs(point) > 1e-9)
    if not usable.any():
        raise ValueError("no usable calibration residuals")
    relative_errors = np.abs(actual[usable] - point[usable]) / np.abs(point[usable])
    radius = float(np.quantile(relative_errors, coverage, method="higher"))
    result = prediction.copy()
    lower = result["prediction_q50"] * max(0.0, 1.0 - radius)
    upper = result["prediction_q50"] * (1.0 + radius)
    result["prediction_q10"] = np.minimum(result["prediction_q10"], lower)
    result["prediction_q90"] = np.maximum(result["prediction_q90"], upper)
    return result, radius


def apply_regional_relative_split_conformal_interval(
    prediction: pd.DataFrame,
    calibration: pd.DataFrame,
    actual_column: str,
    point_column: str,
    coverage: float = 0.8,
    min_region_rows: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate relative conformal radii by region, with explicit global fallback.

    A shared radius obscures regional heterogeneity.  Each region with enough
    usable calibration observations obtains its own finite-sample radius; a
    smaller region falls back to the global radius and is labelled as such.
    """
    required_prediction = {"region_code", "prediction_q10", "prediction_q50", "prediction_q90"}
    required_calibration = {"region_code", actual_column, point_column}
    if required_prediction.difference(prediction.columns) or required_calibration.difference(calibration.columns):
        raise ValueError("regional conformal inputs are incomplete")
    if min_region_rows < 1:
        raise ValueError("min_region_rows must be positive")
    _, global_radius = apply_relative_split_conformal_interval(
        prediction,
        calibration_actual=calibration[actual_column],
        calibration_point=calibration[point_column],
        coverage=coverage,
    )
    result = prediction.copy()
    records: list[dict[str, object]] = []
    regions = sorted(result["region_code"].astype(str).unique())
    for region in regions:
        subset = calibration.loc[calibration["region_code"].astype(str).eq(region), [actual_column, point_column]]
        actual = pd.to_numeric(subset[actual_column], errors="coerce")
        point = pd.to_numeric(subset[point_column], errors="coerce")
        usable = actual.notna() & point.notna() & point.abs().gt(1e-9)
        usable_rows = int(usable.sum())
        if usable_rows >= min_region_rows:
            _, radius = apply_relative_split_conformal_interval(
                result.loc[result["region_code"].astype(str).eq(region)],
                calibration_actual=actual.loc[usable],
                calibration_point=point.loc[usable],
                coverage=coverage,
            )
            status = "region_specific"
        else:
            radius = global_radius
            status = "global_fallback_insufficient_region_calibration"
        mask = result["region_code"].astype(str).eq(region)
        lower = result.loc[mask, "prediction_q50"] * max(0.0, 1.0 - radius)
        upper = result.loc[mask, "prediction_q50"] * (1.0 + radius)
        result.loc[mask, "prediction_q10"] = np.minimum(result.loc[mask, "prediction_q10"], lower)
        result.loc[mask, "prediction_q90"] = np.maximum(result.loc[mask, "prediction_q90"], upper)
        records.append({"region_code": region, "relative_radius": float(radius), "calibration_rows": usable_rows, "calibration_status": status, "global_fallback_radius": float(global_radius)})
    return result, pd.DataFrame(records)


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
    region_specific = "region_code" in source.columns
    if region_specific:
        regions = pd.DataFrame({"region_code": sorted(source["region_code"].astype(str).unique())})
        dates["_key"] = 1; regions["_key"] = 1
        dates = dates.merge(regions, on="_key", how="inner").drop(columns="_key")
    group_columns = (["region_code"] if region_specific else []) + ["month", "day"]
    profile = source.groupby(group_columns, as_index=False)[feature_columns].median()
    dates["month"] = dates["date"].dt.month; dates["day"] = dates["date"].dt.day
    result = dates.merge(profile, on=group_columns, how="left")
    monthly_group = (["region_code"] if region_specific else []) + ["month"]
    monthly = source.groupby(monthly_group, as_index=False)[feature_columns].median()
    result = result.merge(monthly, on=monthly_group, how="left", suffixes=("", "_monthly"))
    for column in feature_columns:
        result[column] = result[column].fillna(result[f"{column}_monthly"]).fillna(float(source[column].median()))
    result = result.drop(columns=[f"{column}_monthly" for column in feature_columns])
    return result.drop(columns=["month", "day"])


def select_regional_search_lags(
    observed: pd.DataFrame,
    search_features: pd.DataFrame,
    min_rows: int = 20,
) -> pd.DataFrame:
    """Choose a destination-search lag per region from observed pre-forecast dates.

    The selector is deliberately descriptive: it uses only the sparse, real
    attraction-index observations available before the 2026 forecast year and
    records its sample size and rank correlation instead of claiming a causal
    demand effect.
    """
    required_observed = {"date", "region_code", "visitor_index"}
    if required_observed.difference(observed.columns) or "date" not in search_features.columns:
        raise ValueError("observed and search features do not contain required columns")
    candidates = [
        column for column in [
            "theme_destination_lag_1", "theme_destination_lag_2",
            "theme_destination_lag_3", "theme_destination_lag_7",
        ] if column in search_features.columns
    ]
    if not candidates:
        raise ValueError("search features contain no supported destination-search lags")
    source = search_features.loc[:, ["date", *candidates]].copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    panel = observed.loc[:, ["date", "region_code", "visitor_index"]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    panel["visitor_index"] = pd.to_numeric(panel["visitor_index"], errors="coerce")
    panel = panel.dropna(subset=["visitor_index"]).groupby(["date", "region_code"], as_index=False)["visitor_index"].median()
    merged = panel.merge(source, on="date", how="left")
    lag_order = {column: int(column.rsplit("_", 1)[-1]) for column in candidates}
    rows: list[dict[str, object]] = []
    for region, group in merged.groupby("region_code", sort=True):
        scores = []
        for column in candidates:
            valid = group.loc[:, ["visitor_index", column]].dropna()
            has_variation = valid["visitor_index"].nunique() > 1 and valid[column].nunique() > 1
            correlation = valid["visitor_index"].corr(valid[column], method="spearman") if len(valid) >= min_rows and has_variation else np.nan
            scores.append((column, int(len(valid)), float(correlation) if pd.notna(correlation) else np.nan))
        eligible = [score for score in scores if score[1] >= min_rows and np.isfinite(score[2])]
        if eligible:
            selected, rows_used, correlation = sorted(eligible, key=lambda item: (-abs(item[2]), lag_order[item[0]]))[0]
            status = "selected_from_observed_training_dates"
        else:
            selected = min(candidates, key=lambda column: lag_order[column])
            rows_used = next(score[1] for score in scores if score[0] == selected)
            correlation = next(score[2] for score in scores if score[0] == selected)
            status = "fallback_shortest_lag_insufficient_observations"
        rows.append({"region_code": str(region), "selected_search_column": selected, "selected_lag_days": lag_order[selected], "observed_rows": rows_used, "spearman_correlation": correlation, "selection_status": status})
    return pd.DataFrame(rows)


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
    join_columns = ["date"]
    if "region_code" in source.columns:
        if "region_code" not in result.columns:
            raise ValueError("region-specific covariates require region_code in target frame")
        join_columns.append("region_code")
    columns = [*join_columns, *[column for column in ["rain_mm", "temperature_c", "search_lag_1"] if column in source.columns]]
    return result.merge(source.loc[:, columns].drop_duplicates(join_columns), on=join_columns, how="left")


def _daily_comparison(regional: pd.DataFrame, observed: pd.DataFrame, covariates: pd.DataFrame | None = None, min_holdout_rows: int = 50) -> tuple[pd.DataFrame, bool]:
    observed_frame = observed.loc[:, ["date", "region_code", "visitor_index"]].copy()
    observed_frame["date"] = pd.to_datetime(observed_frame["date"], errors="raise")
    observed_frame["visitor_index"] = pd.to_numeric(observed_frame["visitor_index"], errors="coerce")
    observed_frame = observed_frame.dropna(subset=["visitor_index"])
    regional_dates = pd.to_datetime(regional["date"], errors="raise")
    observed_frame = observed_frame.loc[
        observed_frame["region_code"].astype(str).isin(regional["region_code"].astype(str).unique())
        & observed_frame["date"].between(regional_dates.min(), regional_dates.max())
    ].copy()
    observed_panel = observed_frame.groupby(["date", "region_code"], as_index=False)["visitor_index"].median()
    if observed_panel.empty:
        return pd.DataFrame(columns=["candidate", "test_rows", "mae", "rmse", "smape"]), True
    year_counts = observed_panel.groupby(observed_panel["date"].dt.year).size()
    eligible_years = year_counts.loc[year_counts.ge(min_holdout_rows)]
    if eligible_years.empty:
        return pd.DataFrame(columns=["candidate", "test_rows", "mae", "rmse", "smape", "picp_80", "mean_interval_width", "holdout_year"]), True
    holdout_year = int(eligible_years.index.max())
    first_test_date = pd.Timestamp(f"{holdout_year}-01-01")
    observed_frame = observed_panel.loc[observed_panel["date"].dt.year.eq(holdout_year)].copy()
    train = regional.loc[pd.to_datetime(regional["date"], errors="raise") < first_test_date].copy()
    train = _attach_covariates(train, covariates)
    if len(train) < 3:
        return pd.DataFrame(columns=["candidate", "test_rows", "mae", "rmse", "smape"]), True
    features = _attach_covariates(observed_frame.loc[:, ["date", "region_code"]], covariates)
    calibration = regional.copy(); calibration["date"] = pd.to_datetime(calibration["date"], errors="raise")
    calibration = calibration.loc[calibration["date"] < first_test_date].merge(
        observed_panel, on=["date", "region_code"], how="inner"
    )
    calibration["pressure_index"] = pd.to_numeric(calibration["pressure_index"], errors="coerce")
    calibration["visitor_index"] = pd.to_numeric(calibration["visitor_index"], errors="coerce")
    calibration = calibration.loc[calibration["pressure_index"].gt(0) & calibration["visitor_index"].notna()]
    scales = calibration.groupby("region_code").apply(lambda x: float(np.median(x["visitor_index"] / x["pressure_index"])), include_groups=False).to_dict()
    rows = []
    for name, dynamic in [("seasonal_calendar_ridge", False), ("dynamic_ridge", True)]:
        model = fit_daily_ridge_forecaster(train, "pressure_index", dynamic=dynamic)
        predicted = predict_daily_ridge_forecaster(model, features)
        merged = observed_frame.merge(predicted, on=["date", "region_code"], how="inner")
        scale = merged["region_code"].map(scales).fillna(1.0)
        for q in ["q10", "q50", "q90"]:
            merged[f"prediction_{q}"] *= scale
        if merged.empty:
            continue
        calibration_features = _attach_covariates(calibration.loc[:, ["date", "region_code"]], covariates)
        calibration_prediction = calibration.loc[:, ["date", "region_code", "visitor_index"]].merge(
            predict_daily_ridge_forecaster(model, calibration_features), on=["date", "region_code"], how="inner"
        )
        calibration_scale = calibration_prediction["region_code"].map(scales).fillna(1.0)
        calibration_prediction["prediction_q50"] *= calibration_scale
        merged, regional_radii = apply_regional_relative_split_conformal_interval(
            merged,
            calibration_prediction,
            actual_column="visitor_index",
            point_column="prediction_q50",
        )
        radius_map = dict(zip(regional_radii["region_code"], regional_radii["relative_radius"]))
        row = {"candidate": name, "test_rows": int(len(merged)), "holdout_year": holdout_year, **_score(merged["visitor_index"], merged["prediction_q50"]), "picp_80": float(((merged["visitor_index"] >= merged["prediction_q10"]) & (merged["visitor_index"] <= merged["prediction_q90"])).mean()), "mean_interval_width": float((merged["prediction_q90"] - merged["prediction_q10"]).mean()), "interval_calibration": f"regional_relative_split_conformal_pre_{holdout_year}", "conformal_relative_radius": float(np.median(regional_radii["relative_radius"])), "conformal_relative_radius_by_region": json.dumps(radius_map, ensure_ascii=False, sort_keys=True), "conformal_calibration_by_region": regional_radii.to_json(orient="records", force_ascii=False), "calibration_rows": int(len(calibration_prediction))}
        rows.append(row)
    weekday_median = train.copy(); weekday_median["weekday"] = pd.to_datetime(weekday_median["date"]).dt.dayofweek
    test_naive = observed_frame.copy(); test_naive["weekday"] = test_naive["date"].dt.dayofweek
    historical = calibration.copy(); historical["weekday"] = historical["date"].dt.dayofweek
    medians = historical.groupby(["region_code", "weekday"])["visitor_index"].median()
    test_naive["prediction"] = [medians.get((row.region_code, row.weekday), historical["visitor_index"].median()) for row in test_naive.itertuples()]
    if not test_naive.empty:
        rows.append({"candidate": "historical_weekday_median", "test_rows": int(len(test_naive)), "holdout_year": holdout_year, **_score(test_naive["visitor_index"], test_naive["prediction"]), "picp_80": np.nan, "mean_interval_width": np.nan, "interval_calibration": "not_applicable", "conformal_relative_radius": np.nan, "conformal_relative_radius_by_region": "", "conformal_calibration_by_region": "", "calibration_rows": int(len(calibration))})
    comparison = pd.DataFrame(rows)
    return comparison, comparison.empty


def daily_rolling_origin_validation(
    regional: pd.DataFrame,
    observed: pd.DataFrame,
    covariates: pd.DataFrame | None = None,
    origins: Sequence[str | pd.Timestamp] = (),
    horizon_days: int = 92,
    min_test_rows: int = 20,
) -> pd.DataFrame:
    """Evaluate daily candidates on forward, fixed-horizon observation windows.

    Each origin is strictly causal: model fitting, anchor scaling, and conformal
    calibration use only dates before the origin.  Weather and lagged-search
    covariates in a held-out window are retained for conditional validation,
    which is recorded in the output rather than presented as an ex-ante test.
    """
    if horizon_days < 1 or min_test_rows < 1:
        raise ValueError("horizon_days and min_test_rows must be positive")
    required_regional = {"date", "region_code", "pressure_index"}
    required_observed = {"date", "region_code", "visitor_index"}
    if required_regional.difference(regional.columns) or required_observed.difference(observed.columns):
        raise ValueError("regional and observed tables do not contain required columns")
    columns = [
        "candidate", "origin_date", "window_start", "window_end", "training_end", "calibration_end",
        "evaluation_scope", "region_code", "test_rows", "mae", "rmse", "smape", "wape", "picp_80",
        "mean_interval_width", "interval_calibration", "conformal_relative_radius",
        "conformal_relative_radius_by_region", "conformal_calibration_by_region", "calibration_rows",
        "validation_covariate_mode",
    ]
    regional_frame = regional.copy()
    regional_frame["date"] = pd.to_datetime(regional_frame["date"], errors="raise")
    observed_frame = observed.loc[:, ["date", "region_code", "visitor_index"]].copy()
    observed_frame["date"] = pd.to_datetime(observed_frame["date"], errors="raise")
    observed_frame["visitor_index"] = pd.to_numeric(observed_frame["visitor_index"], errors="coerce")
    observed_panel = observed_frame.dropna(subset=["visitor_index"])
    observed_panel = observed_panel.loc[
        observed_panel["region_code"].astype(str).isin(regional_frame["region_code"].astype(str).unique())
        & observed_panel["date"].between(regional_frame["date"].min(), regional_frame["date"].max())
    ].groupby(["date", "region_code"], as_index=False)["visitor_index"].median()
    records: list[dict[str, object]] = []
    for raw_origin in origins:
        origin = pd.Timestamp(raw_origin).normalize()
        window_end = origin + pd.Timedelta(days=horizon_days - 1)
        train = regional_frame.loc[regional_frame["date"] < origin].copy()
        test = observed_panel.loc[observed_panel["date"].between(origin, window_end)].copy()
        calibration = regional_frame.loc[regional_frame["date"] < origin].merge(
            observed_panel, on=["date", "region_code"], how="inner"
        )
        calibration["pressure_index"] = pd.to_numeric(calibration["pressure_index"], errors="coerce")
        calibration["visitor_index"] = pd.to_numeric(calibration["visitor_index"], errors="coerce")
        calibration = calibration.loc[calibration["pressure_index"].gt(0) & calibration["visitor_index"].notna()].copy()
        if len(train) < 3 or len(test) < min_test_rows or calibration.empty:
            continue
        scales = calibration.groupby("region_code").apply(
            lambda group: float(np.median(group["visitor_index"] / group["pressure_index"])),
            include_groups=False,
        ).to_dict()
        train_features = _attach_covariates(train, covariates)
        test_features = _attach_covariates(test.loc[:, ["date", "region_code"]], covariates)
        calibration_features = _attach_covariates(calibration.loc[:, ["date", "region_code"]], covariates)
        for candidate, dynamic in [("seasonal_calendar_ridge", False), ("dynamic_ridge", True)]:
            model = fit_daily_ridge_forecaster(train_features, "pressure_index", dynamic=dynamic)
            prediction = predict_daily_ridge_forecaster(model, test_features)
            merged = test.merge(prediction, on=["date", "region_code"], how="inner")
            if merged.empty:
                continue
            scale = merged["region_code"].map(scales).fillna(1.0)
            for quantile in ["q10", "q50", "q90"]:
                merged[f"prediction_{quantile}"] *= scale
            calibration_prediction = calibration.loc[:, ["date", "region_code", "visitor_index"]].merge(
                predict_daily_ridge_forecaster(model, calibration_features), on=["date", "region_code"], how="inner"
            )
            calibration_prediction["prediction_q50"] *= calibration_prediction["region_code"].map(scales).fillna(1.0)
            merged, regional_radii = apply_regional_relative_split_conformal_interval(
                merged,
                calibration_prediction,
                actual_column="visitor_index",
                point_column="prediction_q50",
            )
            radius_map = dict(zip(regional_radii["region_code"], regional_radii["relative_radius"]))
            scopes: list[tuple[str, str, pd.DataFrame]] = [("all_dates_all_regions", "ALL", merged)]
            scopes.extend((f"region__{region}", str(region), group) for region, group in merged.groupby("region_code", sort=True))
            summer = merged.loc[merged["date"].dt.month.isin([7, 8])]
            if not summer.empty:
                scopes.append(("summer_all_regions", "ALL", summer))
                scopes.extend((f"summer__{region}", str(region), group) for region, group in summer.groupby("region_code", sort=True))
            for scope, region_code, group in scopes:
                if len(group) < min_test_rows:
                    continue
                scores = _score(group["visitor_index"], group["prediction_q50"])
                scores["wape"] = float(100.0 * np.abs(group["visitor_index"] - group["prediction_q50"]).sum() / max(np.abs(group["visitor_index"]).sum(), 1e-9))
                records.append({
                    "candidate": candidate, "origin_date": origin, "window_start": origin, "window_end": window_end,
                    "training_end": train["date"].max(), "calibration_end": calibration["date"].max(),
                    "evaluation_scope": scope, "region_code": region_code, "test_rows": int(len(group)), **scores,
                    "picp_80": float(((group["visitor_index"] >= group["prediction_q10"]) & (group["visitor_index"] <= group["prediction_q90"])).mean()),
                    "mean_interval_width": float((group["prediction_q90"] - group["prediction_q10"]).mean()),
                    "interval_calibration": f"regional_relative_split_conformal_pre_{origin.date()}",
                    "conformal_relative_radius": float(np.median(regional_radii["relative_radius"])),
                    "conformal_relative_radius_by_region": json.dumps(radius_map, ensure_ascii=False, sort_keys=True),
                    "conformal_calibration_by_region": regional_radii.to_json(orient="records", force_ascii=False),
                    "calibration_rows": int(len(calibration_prediction)),
                    "validation_covariate_mode": "conditional_on_realized_weather_and_lagged_search" if covariates is not None else "calendar_only",
                })
    return pd.DataFrame(records, columns=columns)


def build_q2_diagnostic_figures(monthly: pd.DataFrame, daily: pd.DataFrame, output_directory: str | Path) -> None:
    """Render two compact paper figures from forecast tables, never as observed counts."""
    output = Path(output_directory); output.mkdir(parents=True, exist_ok=True)
    labels = q2_paper_figure_labels()
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    axis.bar(monthly["month"], monthly["visitor_estimate"], color="#4C78A8")
    axis.set(xlabel=labels["monthly_x"], ylabel=labels["monthly_y"], title=labels["monthly_title"])
    if {"annual_forecast_lower", "annual_forecast_upper"}.issubset(monthly.columns):
        interval = f"年度 90% 预测区间：{monthly['annual_forecast_lower'].iloc[0]:,.0f}–{monthly['annual_forecast_upper'].iloc[0]:,.0f}"
        axis.text(0.02, 0.95, interval, transform=axis.transAxes, va="top")
    figure.tight_layout(); figure.savefig(output / "q2_city_monthly_forecast.png", dpi=220); plt.close(figure)
    frame = daily.copy(); frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.loc[frame["date"].dt.month.isin([7, 8])]
    figure, axis = plt.subplots(figsize=(7.2, 3.6))
    region_labels = {"BDH": "北戴河", "HGA": "海港--阿那亚", "SHG": "山海关"}
    for region, group in frame.groupby("region_code"):
        group = group.sort_values("date")
        axis.plot(group["date"], group["visitor_estimate_q50"], label=region_labels.get(str(region), str(region)))
        axis.fill_between(group["date"], group["visitor_estimate_q10"], group["visitor_estimate_q90"], alpha=0.16)
    axis.set(xlabel=labels["summer_x"], ylabel=labels["summer_y"], title=labels["summer_title"])
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
    rolling_validation = daily_rolling_origin_validation(
        regional,
        observed,
        daily_covariates,
        origins=("2024-01-01", "2024-04-01", "2024-07-01"),
        horizon_days=92,
        min_test_rows=20,
    )
    rolling_validation.to_csv(output / "q2_daily_rolling_origin_validation.csv", index=False, encoding="utf-8-sig")
    model_comparison = comparison.loc[comparison["candidate"].isin(["seasonal_calendar_ridge", "dynamic_ridge"])]
    selected_validation = model_comparison.sort_values("mae").iloc[0] if not model_comparison.empty else None
    selected_dynamic = bool(selected_validation["candidate"] == "dynamic_ridge") if selected_validation is not None else True
    conformal_radius = float(selected_validation["conformal_relative_radius"]) if selected_validation is not None and pd.notna(selected_validation.get("conformal_relative_radius")) else np.nan
    regional_radius_map = json.loads(str(selected_validation["conformal_relative_radius_by_region"])) if selected_validation is not None and pd.notna(selected_validation.get("conformal_relative_radius_by_region")) and str(selected_validation["conformal_relative_radius_by_region"]).strip() else {}
    regional_calibration_records = json.loads(str(selected_validation["conformal_calibration_by_region"])) if selected_validation is not None and pd.notna(selected_validation.get("conformal_calibration_by_region")) and str(selected_validation["conformal_calibration_by_region"]).strip() else []
    interval_calibration = str(selected_validation["interval_calibration"]) if selected_validation is not None and pd.notna(selected_validation.get("interval_calibration")) else "model_quantile_only_no_calibration_observations"
    train = regional.loc[pd.to_datetime(regional["date"], errors="raise").dt.year < forecast_year].copy()
    train = _attach_covariates(train, daily_covariates)
    daily_model = fit_daily_ridge_forecaster(train, "pressure_index", dynamic=selected_dynamic)
    regions = sorted(train["region_code"].astype(str).unique())
    future_dates = pd.date_range(f"{forecast_year}-01-01", f"{forecast_year}-12-31", freq="D")
    future = pd.DataFrame({"date": np.repeat(future_dates, len(regions)), "region_code": regions * len(future_dates)})
    future_covariates = build_climatological_covariates(daily_covariates, forecast_year) if daily_covariates is not None else None
    future = _attach_covariates(future, future_covariates)
    daily = predict_daily_ridge_forecaster(daily_model, future)
    if regional_radius_map:
        radii = daily["region_code"].map(regional_radius_map).fillna(conformal_radius)
        lower = daily["prediction_q50"] * np.maximum(0.0, 1.0 - radii)
        upper = daily["prediction_q50"] * (1.0 + radii)
        daily["prediction_q10"] = np.minimum(daily["prediction_q10"], lower)
        daily["prediction_q90"] = np.maximum(daily["prediction_q90"], upper)
    elif np.isfinite(conformal_radius):
        lower = daily["prediction_q50"] * max(0.0, 1.0 - conformal_radius)
        upper = daily["prediction_q50"] * (1.0 + conformal_radius)
        daily["prediction_q10"] = np.minimum(daily["prediction_q10"], lower)
        daily["prediction_q90"] = np.maximum(daily["prediction_q90"], upper)
    daily["interval_calibration"] = interval_calibration
    daily["conformal_relative_radius"] = daily["region_code"].map(regional_radius_map).fillna(conformal_radius)
    calibration_table = pd.DataFrame(regional_calibration_records)
    if not calibration_table.empty:
        calibration_table["calibration_method"] = interval_calibration
    calibration_table.to_csv(output / f"q2_daily_conformal_calibration_{forecast_year}.csv", index=False, encoding="utf-8-sig")
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
    rainy = scenario_adjust_weather(future, rain_add_mm=20.0)
    rain_daily = predict_daily_ridge_forecaster(weather_model, rainy).merge(scales, on="region_code", how="left")
    scenario = daily.loc[:, ["date", "region_code", "visitor_estimate_q50"]].rename(columns={"visitor_estimate_q50": "baseline_estimate"})
    scenario["weather_response_baseline_estimate"] = weather_base["prediction_q50"].to_numpy() * weather_base["baseline_scale"].to_numpy()
    scenario["rain_shock_estimate"] = rain_daily["prediction_q50"].to_numpy() * rain_daily["baseline_scale"].to_numpy()
    heat = scenario_adjust_weather(future, temperature_shift_c=5.0)
    heat_daily = predict_daily_ridge_forecaster(weather_model, heat).merge(scales, on="region_code", how="left")
    scenario["heat_shock_estimate"] = heat_daily["prediction_q50"].to_numpy() * heat_daily["baseline_scale"].to_numpy()
    scenario["scenario_label"] = "additive_20mm_rainfall_and_plus5c_weather_shocks"
    scenario.to_csv(output / f"q2_weather_shock_{forecast_year}.csv", index=False, encoding="utf-8-sig")
    rain_shock_effective_share = float(
        1.0 - np.isclose(
            scenario["weather_response_baseline_estimate"],
            scenario["rain_shock_estimate"],
        ).mean()
    )
    build_q2_diagnostic_figures(monthly, daily, output)
    report = {
        "forecast_year": forecast_year,
        "annual_forecast_point": annual_forecast["point"],
        "annual_forecast_interval": [annual_forecast["lower"], annual_forecast["upper"]],
        "annual_selected_model": annual_selected_model,
        "daily_selected_model": "dynamic_ridge" if selected_dynamic else "seasonal_calendar_ridge",
        "daily_interval_calibration": interval_calibration,
        "daily_conformal_relative_radius": None if not np.isfinite(conformal_radius) else conformal_radius,
        "daily_conformal_relative_radius_by_region": regional_radius_map,
        "daily_validation_scope": "raw_attraction_observation_dates_only" if not no_independent_observation else "no_independent_observation_available",
        "daily_validation_covariate_mode": "conditional_on_realized_weather_and_lagged_search" if daily_covariates is not None else "calendar_only",
        "daily_rolling_origin_windows": ["2024-01-01", "2024-04-01", "2024-07-01"],
        "daily_rolling_validation_rows": int(len(rolling_validation)),
        "weather_response_model": weather_response_model,
        "weather_shock_effective": bool(rain_shock_effective_share > 0.0),
        "rain_shock_effective_share": rain_shock_effective_share,
        "regional_daily_unit": "anchor_constrained_visitor_scale_estimate_not_observed_count",
    }
    (output / "q2_quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    prepare_submission_runtime(Path(__file__).resolve().parents[1])


"""Run the Question 2 two-scale forecast pipeline."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]





annual = pd.read_csv(ROOT / "data/runtime/clean/annual_city_tourism_1999_2024.csv", encoding="utf-8-sig")
annual = annual.loc[annual["\u6307\u6807"].eq("\u56fd\u5185\u65c5\u6e38\u4eba\u6b21"), ["year", "\u6570\u503c", "\u5355\u4f4d"]].rename(columns={"\u6570\u503c": "official_total"})
if not annual["\u5355\u4f4d"].eq("\u5343\u4eba\u6b21").all():
    raise ValueError("official annual unit must be thousand visitor-trips before conversion")
annual["official_total"] = annual["official_total"] * 1000.0
regional = pd.read_csv(ROOT / "data/runtime/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv", encoding="utf-8-sig")
observed = pd.read_csv(ROOT / "data/runtime/model_input/censored_attraction_observation_panel.csv", encoding="utf-8-sig")
weather = pd.read_csv(ROOT / "data/runtime/clean/daily_weather_2015_2025.csv", encoding="utf-8-sig")
weather = weather.loc[:, ["date", "\u65e5\u5747\u6c14\u6e29_\u2103", "\u65e5\u964d\u6c34\u91cf_mm"]].rename(columns={"\u65e5\u5747\u6c14\u6e29_\u2103": "temperature_c", "\u65e5\u964d\u6c34\u91cf_mm": "rain_mm"})
output = ROOT / "outputs/runtime/question2_analysis"
output.mkdir(parents=True, exist_ok=True)
search = pd.read_csv(ROOT / "data/runtime/model_input/daily_search_theme_features_2016_2026.csv", encoding="utf-8-sig")
model_regions = set(regional["region_code"].astype(str).unique())
lag_selection = select_regional_search_lags(observed.loc[observed["region_code"].astype(str).isin(model_regions)], search)
lag_selection.to_csv(output / "q2_regional_search_lag_selection.csv", index=False, encoding="utf-8-sig")
search_columns = ["date", "theme_destination_lag_1", "theme_destination_lag_2", "theme_destination_lag_3", "theme_destination_lag_7"]
search = search.loc[:, [column for column in search_columns if column in search.columns]]
base_covariates = weather.merge(search, on="date", how="outer")
regional_covariates = []
for row in lag_selection.itertuples(index=False):
    part = base_covariates.loc[:, ["date", "temperature_c", "rain_mm", row.selected_search_column]].copy()
    part = part.rename(columns={row.selected_search_column: "search_lag_1"})
    part["region_code"] = row.region_code
    regional_covariates.append(part)
covariates = pd.concat(regional_covariates, ignore_index=True)
print(build_question2_outputs(annual, regional, observed, output, daily_covariates=covariates))
anchors = pd.read_csv(ROOT / "data/runtime/clean/raw_cleaned/qinhuangdao_summer_regional_flow_anchors_2023_2025.csv", encoding="utf-8-sig")
city_daily = pd.read_csv(ROOT / "outputs/runtime/question1_analysis/q1_city_daily_annual_constrained.csv", encoding="utf-8-sig")
validation = build_external_anchor_validation(anchors, city_daily, regional)
validation.to_csv(output / "q2_external_anchor_validation.csv", index=False, encoding="utf-8-sig")

if __name__ == "__main__":
    _export_submission(ROOT, "Q2_2026_forecast.csv", [
        "outputs/runtime/question2_analysis/q2_city_monthly_forecast_2026.csv",
        "outputs/runtime/question2_analysis/q2_region_daily_forecast_2026.csv",
    ])
    _export_submission(ROOT, "Q2_validation_summary.csv", [
        "outputs/runtime/question2_analysis/q2_annual_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_daily_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_daily_rolling_origin_validation.csv",
        "outputs/runtime/question2_analysis/q2_daily_conformal_calibration_2026.csv",
        "outputs/runtime/question2_analysis/q2_external_anchor_validation.csv",
    ])
