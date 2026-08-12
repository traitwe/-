# question2_core.py: 以下函数从完整运行程序的真实实现原样摘取。

# 完整输入、输出与辅助函数见支撑材料中的 question*_model.py。

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
