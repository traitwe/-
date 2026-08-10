"""Transparent two-state regularized baseline for relative tourism pressure."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _calendar_flag(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0, index=frame.index, dtype=int)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0).clip(0, 1)


def assign_calendar_regime(calendar: pd.DataFrame) -> pd.DataFrame:
    """Label high-pressure days from exogenous summer and statutory-holiday information."""
    if "date" not in calendar.columns:
        raise ValueError("calendar missing column: date")
    result = calendar.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    holiday = _calendar_flag(result, "是否法定放假")
    result["is_summer"] = result["date"].dt.month.isin([7, 8]).astype(int)
    result["is_holiday"] = holiday.clip(lower=0, upper=1)
    result["is_weekend"] = _calendar_flag(result, "是否周末")
    result["regime"] = np.where(
        (result["is_summer"] == 1) | (result["is_holiday"] == 1),
        "high_pressure",
        "normal",
    )
    return result


def fit_pressure_baseline(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    weather: pd.DataFrame,
    search_features: pd.DataFrame,
    ridge_alpha: float = 10.0,
    prediction_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Fit a ridge baseline to observed relative indices and predict sampled region-days."""
    required = {"date", "region_code", "is_observed", "visitor_index"}
    if missing := required.difference(panel.columns):
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if ridge_alpha < 0:
        raise ValueError("ridge_alpha must be non-negative")

    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["visitor_index"] = pd.to_numeric(raw["visitor_index"], errors="coerce")
    observed = raw.loc[raw["is_observed"] & raw["visitor_index"].notna()]
    targets = (
        observed.groupby(["date", "region_code"], as_index=False)["visitor_index"]
        .median()
        .rename(columns={"visitor_index": "observed_visitor_index"})
    )
    training_pairs = raw.loc[:, ["date", "region_code"]].drop_duplicates()
    if prediction_frame is None:
        output_pairs = training_pairs.copy()
    else:
        prediction_required = {"date", "region_code"}
        if missing := prediction_required.difference(prediction_frame.columns):
            raise ValueError(f"prediction_frame missing columns: {sorted(missing)}")
        output_pairs = prediction_frame.loc[:, ["date", "region_code"]].copy()
        output_pairs["date"] = pd.to_datetime(output_pairs["date"], errors="coerce")
        output_pairs = output_pairs.dropna(subset=["date", "region_code"]).drop_duplicates()
    result = pd.concat(
        [training_pairs.assign(_fit_role="train"), output_pairs.assign(_fit_role="output")],
        ignore_index=True,
    ).merge(targets, on=["date", "region_code"], how="left")
    calendar_features = assign_calendar_regime(calendar)
    result = result.merge(
        calendar_features.loc[:, ["date", "is_summer", "is_holiday", "is_weekend", "regime"]],
        on="date",
        how="left",
    )

    weather = weather.copy()
    weather["date"] = pd.to_datetime(weather["date"], errors="coerce")
    weather_columns = [name for name in ["日均气温_℃", "日降水量_mm"] if name in weather.columns]
    if weather_columns:
        result = result.merge(weather.loc[:, ["date", *weather_columns]], on="date", how="left")
    search = search_features.copy()
    search["date"] = pd.to_datetime(search["date"], errors="coerce")
    search_columns = [name for name in search.columns if name.startswith("theme_") and name.endswith("_lag_1")]
    if search_columns:
        result = result.merge(search.loc[:, ["date", *search_columns]], on="date", how="left")

    numeric_columns = ["is_summer", "is_holiday", "is_weekend", *weather_columns, *search_columns]
    for column in numeric_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
        result[column] = result[column].fillna(result[column].median()).fillna(0.0)
    region_dummies = pd.get_dummies(result["region_code"], prefix="region", dtype=float)
    feature_frame = pd.concat(
        [pd.Series(1.0, index=result.index, name="intercept"), result.loc[:, numeric_columns], region_dummies],
        axis=1,
    )
    target_mask = result["_fit_role"].eq("train") & result["observed_visitor_index"].notna()
    if not target_mask.any():
        raise ValueError("no observed visitor-index rows available for baseline fitting")
    x = feature_frame.to_numpy(dtype=float)
    train_x = x[target_mask.to_numpy()]
    train_y = np.log1p(result.loc[target_mask, "observed_visitor_index"].clip(lower=0).to_numpy(dtype=float))
    penalty = np.eye(train_x.shape[1]) * ridge_alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.pinv(train_x.T @ train_x + penalty) @ train_x.T @ train_y
    prediction_log = x @ coefficients
    result["pressure_index"] = np.maximum(np.expm1(prediction_log), 0.0)
    residuals = train_y - train_x @ coefficients
    result["uncertainty_proxy"] = float(np.std(residuals, ddof=0))
    result["estimate_label"] = "relative_pressure_index"
    result = result.loc[result["_fit_role"].eq("output")].drop(columns="_fit_role")
    return result.sort_values(["date", "region_code"]).reset_index(drop=True)


def rolling_time_validation(
    panel: pd.DataFrame,
    calendar: pd.DataFrame,
    weather: pd.DataFrame,
    search_features: pd.DataFrame,
    cutoffs: list[str],
    ridge_alpha: float = 10.0,
) -> pd.DataFrame:
    """Evaluate forward predictions using only observations available before each cutoff."""
    raw = panel.copy()
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["visitor_index"] = pd.to_numeric(raw["visitor_index"], errors="coerce")
    actual = (
        raw.loc[raw["is_observed"] & raw["visitor_index"].notna()]
        .groupby(["date", "region_code"], as_index=False)["visitor_index"]
        .median()
        .rename(columns={"visitor_index": "actual_visitor_index"})
    )
    scores: list[dict[str, object]] = []
    for cutoff_value in cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        train = raw.loc[raw["date"] <= cutoff]
        test = actual.loc[actual["date"] > cutoff]
        if train.empty or test.empty:
            continue
        prediction = fit_pressure_baseline(
            train,
            calendar,
            weather,
            search_features,
            ridge_alpha=ridge_alpha,
            prediction_frame=test.loc[:, ["date", "region_code"]],
        )
        comparison = prediction.merge(test, on=["date", "region_code"], how="inner")
        scores.append(
            {
                "train_end": cutoff,
                "test_rows": int(len(comparison)),
                "mae": float(
                    (comparison["pressure_index"] - comparison["actual_visitor_index"]).abs().mean()
                ),
            }
        )
    return pd.DataFrame(scores, columns=["train_end", "test_rows", "mae"])
