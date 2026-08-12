"""Runnable final program for one modelling question."""
from pathlib import Path
from common_runtime import _export_submission, prepare_submission_runtime


"""Transparent two-state regularized baseline for relative tourism pressure."""



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


"""Data preparation for the Question 1 hierarchical left-censored pressure model."""



import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import log_ndtr


def prepare_censored_likelihood_data(panel: pd.DataFrame, covariates: pd.DataFrame) -> dict[str, object]:
    """Separate ranked density observations from same-group left-censored counts."""
    required = {"date", "region_code", "attraction_name", "is_observed", "visitor_index"}
    if missing := required.difference(panel.columns):
        raise ValueError(f"panel missing columns: {sorted(missing)}")
    if "date" not in covariates.columns:
        raise ValueError("covariates missing column: date")
    source = panel.copy()
    source["date"] = pd.to_datetime(source["date"], errors="coerce")
    source["visitor_index"] = pd.to_numeric(source["visitor_index"], errors="coerce")
    source["is_observed"] = source["is_observed"].astype("string").str.lower().isin(["true", "1", "yes"])
    features = covariates.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    numeric_features = [
        column for column in features.columns
        if column != "date" and pd.api.types.is_numeric_dtype(features[column])
    ]
    for column in numeric_features:
        features[column] = pd.to_numeric(features[column], errors="coerce")
        features[column] = features[column].fillna(features[column].median())
    source = source.merge(features, on="date", how="left")
    groups = []
    observed = []
    censored = []
    for key, group in source.groupby(["date", "region_code"], sort=True):
        positive = group.loc[group["is_observed"] & group["visitor_index"].gt(0)]
        if positive.empty:
            continue
        group_id = len(groups)
        threshold_log = float(np.log1p(positive["visitor_index"].min()))
        record = {"group_id": group_id, "date": key[0], "region_code": key[1], "censored_count": int((~group["is_observed"]).sum()), "threshold_log": threshold_log}
        record.update({column: float(group.iloc[0][column]) for column in numeric_features})
        groups.append(record)
        for _, row in positive.iterrows():
            observed.append({"group_id": group_id, "attraction_name": row["attraction_name"], "observed_log_index": float(np.log1p(row["visitor_index"]))})
        for _, row in group.loc[~group["is_observed"]].iterrows():
            censored.append({"group_id": group_id, "attraction_name": row["attraction_name"], "threshold_log": threshold_log})
    observed_frame = pd.DataFrame(observed)
    censored_frame = pd.DataFrame(censored)
    group_frame = pd.DataFrame(groups)
    attraction_names = sorted(pd.concat([observed_frame.get("attraction_name", pd.Series(dtype=str)), censored_frame.get("attraction_name", pd.Series(dtype=str))]).dropna().unique().tolist())
    attraction_map = {name: index for index, name in enumerate(attraction_names)}
    if not observed_frame.empty:
        observed_frame["attraction_id"] = observed_frame["attraction_name"].map(attraction_map)
    if not censored_frame.empty:
        censored_frame["attraction_id"] = censored_frame["attraction_name"].map(attraction_map)
    regions = sorted(group_frame["region_code"].unique().tolist()) if not group_frame.empty else []
    region_map = {region: index for index, region in enumerate(regions)}
    if not group_frame.empty:
        group_frame["region_id"] = group_frame["region_code"].map(region_map)
        feature_matrix = group_frame.loc[:, numeric_features].to_numpy(dtype=float) if numeric_features else np.empty((len(group_frame), 0))
        feature_mean = feature_matrix.mean(axis=0) if numeric_features else np.empty(0)
        feature_std = feature_matrix.std(axis=0) if numeric_features else np.empty(0)
        feature_std[feature_std == 0] = 1.0
        standardized_features = (feature_matrix - feature_mean) / feature_std if numeric_features else feature_matrix
    else:
        feature_mean = feature_std = np.empty(0); standardized_features = np.empty((0, len(numeric_features)))
    attraction_region = np.zeros(len(attraction_names), dtype=int)
    if attraction_names:
        attraction_regions = pd.concat([observed_frame[["attraction_name", "group_id"]], censored_frame[["attraction_name", "group_id"]]]).merge(group_frame[["group_id", "region_id"]], on="group_id", how="left").drop_duplicates("attraction_name")
        attraction_region = attraction_regions.set_index("attraction_name").loc[attraction_names, "region_id"].to_numpy(dtype=int)
    return {
        "groups": group_frame,
        "observed": observed_frame,
        "censored": censored_frame,
        "observed_log_index": observed_frame.get("observed_log_index", pd.Series(dtype=float)).to_numpy(),
        "censored_count": group_frame.get("censored_count", pd.Series(dtype=int)).to_numpy(),
        "group_count": len(group_frame),
        "attraction_count": len(attraction_names),
        "region_count": len(regions),
        "region_names": regions,
        "feature_names": numeric_features,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "standardized_features": standardized_features,
        "attraction_region": attraction_region,
    }


def negative_log_posterior(params: dict[str, object], data: dict[str, object], lambda_alpha: float = 0.1) -> float:
    """Evaluate observed normal density plus grouped left-censored normal probability."""
    groups = data["groups"]
    observed = data["observed"]
    eta = np.asarray(params["group_eta"], dtype=float)
    alpha = np.asarray(params["attraction_alpha"], dtype=float)
    sigma = float(np.exp(float(params["log_sigma"])))
    if len(eta) != len(groups) or sigma <= 0:
        return float("inf")
    observed_attraction_id = observed["attraction_id"].to_numpy(dtype=int)
    # The legacy objective predates censored-attraction effects and accepts an
    # observed-attraction-only alpha vector. Keep that public behavior intact.
    if len(alpha) <= observed_attraction_id.max(initial=-1):
        observed_attraction_id = pd.factorize(observed["attraction_name"], sort=True)[0]
    observed_mean = eta[observed["group_id"].to_numpy(dtype=int)] + alpha[observed_attraction_id]
    z = observed["observed_log_index"].to_numpy(dtype=float)
    density_nll = float(np.sum(0.5 * ((z - observed_mean) / sigma) ** 2 + np.log(sigma * np.sqrt(2 * np.pi))))
    threshold = groups["threshold_log"].to_numpy(dtype=float)
    standardized = (threshold - eta) / sigma
    cdf = 0.5 * (1 + np.vectorize(__import__("math").erf)(standardized / np.sqrt(2)))
    censor_nll = float(-np.sum(groups["censored_count"].to_numpy(dtype=float) * np.log(np.maximum(cdf, 1e-12))))
    return density_nll + censor_nll + lambda_alpha * float(np.sum(alpha**2))


def fit_censored_pressure_core(data: dict[str, object]) -> dict[str, object]:
    """Fit sampled-day latent pressures with attraction effects and left-censor likelihood."""
    group_count = int(data["group_count"])
    attraction_count = int(data["attraction_count"])
    initial = np.zeros(group_count + attraction_count + 1)
    initial[-1] = 0.0

    def objective(vector: np.ndarray) -> float:
        params = {"group_eta": vector[:group_count], "attraction_alpha": vector[group_count:-1], "log_sigma": vector[-1]}
        return negative_log_posterior(params, data)

    def gradient(vector: np.ndarray) -> np.ndarray:
        eta, alpha, log_sigma = vector[:group_count], vector[group_count:-1], vector[-1]
        sigma = np.exp(log_sigma); groups, obs = data["groups"], data["observed"]
        gids = obs["group_id"].to_numpy(int); aids = obs["attraction_id"].to_numpy(int); z = obs["observed_log_index"].to_numpy(float)
        residual = eta[gids] + alpha[aids] - z
        g_eta = np.bincount(gids, weights=residual / sigma**2, minlength=group_count)
        g_alpha = np.bincount(aids, weights=residual / sigma**2, minlength=attraction_count) + 0.2 * alpha
        q = (groups["threshold_log"].to_numpy(float) - eta) / sigma
        pdf = np.exp(-0.5*q*q) / np.sqrt(2*np.pi)
        cdf = np.maximum(0.5*(1 + np.vectorize(__import__('math').erf)(q/np.sqrt(2))), 1e-12)
        hazard = pdf / cdf; counts = groups["censored_count"].to_numpy(float)
        g_eta += counts * hazard / sigma
        g_log_sigma = np.sum(1 - (residual/sigma)**2) + np.sum(counts * hazard * q)
        return np.r_[g_eta, g_alpha, g_log_sigma]

    fit = minimize(objective, initial, jac=gradient, method="L-BFGS-B", bounds=[(None, None)] * (len(initial) - 1) + [(-4, 4)], options={"maxiter": 2000, "maxfun": 100000})
    sampled_pressure = data["groups"].loc[:, ["date", "region_code"]].copy()
    sampled_pressure["pressure_index"] = np.maximum(np.expm1(fit.x[:group_count]), 0.0)
    sampled_pressure["estimate_label"] = "censored_likelihood_pressure_index"
    return {"converged": bool(fit.success), "group_eta": fit.x[:group_count], "attraction_alpha": fit.x[group_count:-1], "log_sigma": float(fit.x[-1]), "objective": float(fit.fun), "message": fit.message, "sampled_pressure": sampled_pressure}


def fit_hierarchical_censored_pressure(
    data: dict[str, object], lambda_alpha: float = 0.1, lambda_dynamic: float = 0.25,
    lambda_beta: float = 0.05, lambda_region: float = 0.05,
) -> dict[str, object]:
    """Jointly estimate covariates, region effects and AR-constrained sampled residuals."""
    group_count, attraction_count, region_count = int(data["group_count"]), int(data["attraction_count"]), int(data["region_count"])
    if group_count == 0:
        raise ValueError("no sampled groups available for hierarchical fit")
    x = np.column_stack([np.ones(group_count), np.asarray(data["standardized_features"], dtype=float)])
    feature_names = ["intercept", *list(data["feature_names"])]
    p = x.shape[1]
    groups, observed, censored = data["groups"], data["observed"], data["censored"]
    group_region = groups["region_id"].to_numpy(dtype=int)
    obs_g = observed["group_id"].to_numpy(dtype=int); obs_a = observed["attraction_id"].to_numpy(dtype=int); obs_z = observed["observed_log_index"].to_numpy(dtype=float)
    cen_g = censored.get("group_id", pd.Series(dtype=int)).to_numpy(dtype=int); cen_a = censored.get("attraction_id", pd.Series(dtype=int)).to_numpy(dtype=int); cen_threshold = censored.get("threshold_log", pd.Series(dtype=float)).to_numpy(dtype=float)
    attraction_region = np.asarray(data["attraction_region"], dtype=int)
    previous = np.full(group_count, -1, dtype=int)
    for _, frame in groups.sort_values(["region_code", "date"]).groupby("region_code", sort=False):
        identifiers = frame["group_id"].to_numpy(dtype=int)
        if len(identifiers) > 1:
            previous[identifiers[1:]] = identifiers[:-1]

    core = fit_censored_pressure_core(data)
    target = np.asarray(core["group_eta"], dtype=float)
    design = np.column_stack([x, pd.get_dummies(group_region, dtype=float).to_numpy()[:, 1:]])
    coefficients = np.linalg.pinv(design) @ target
    beta0 = coefficients[:p]; region0 = np.r_[0.0, coefficients[p:]]
    residual0 = target - x @ beta0 - region0[group_region]
    initial = np.r_[beta0, region0[1:], residual0, np.zeros(attraction_count), np.repeat(float(core["log_sigma"]), region_count), 0.0]
    b_start, u_start = p, p + region_count - 1
    a_start, sigma_start, rho_index = u_start + group_count, u_start + group_count + attraction_count, u_start + group_count + attraction_count + region_count

    def unpack(vector: np.ndarray):
        beta = vector[:p]
        regional = np.r_[0.0, vector[b_start:u_start]]
        u = vector[u_start:a_start]
        raw_alpha = vector[a_start:sigma_start]
        alpha = raw_alpha.copy()
        for region_id in range(region_count):
            mask = attraction_region == region_id
            if mask.any(): alpha[mask] -= alpha[mask].mean()
        sigma = np.exp(vector[sigma_start:rho_index])
        rho = np.tanh(vector[rho_index])
        eta = x @ beta + regional[group_region] + u
        return beta, regional, u, raw_alpha, alpha, sigma, rho, eta

    def objective_gradient(vector: np.ndarray):
        beta, regional, u, raw_alpha, alpha, sigma, rho, eta = unpack(vector)
        sigma_group = sigma[group_region]
        residual = eta[obs_g] + alpha[obs_a] - obs_z
        value = float(np.sum(0.5 * (residual / sigma_group[obs_g]) ** 2 + np.log(sigma_group[obs_g] * np.sqrt(2 * np.pi))))
        g_eta = np.bincount(obs_g, weights=residual / sigma_group[obs_g] ** 2, minlength=group_count)
        g_alpha = np.bincount(obs_a, weights=residual / sigma_group[obs_g] ** 2, minlength=attraction_count)
        g_log_sigma = np.bincount(group_region[obs_g], weights=1 - (residual / sigma_group[obs_g]) ** 2, minlength=region_count)
        if len(cen_g):
            q = (cen_threshold - eta[cen_g] - alpha[cen_a]) / sigma_group[cen_g]
            log_cdf = log_ndtr(q)
            value -= float(np.sum(log_cdf))
            log_pdf = -0.5 * q * q - 0.5 * np.log(2 * np.pi)
            hazard = np.exp(np.minimum(log_pdf - log_cdf, 50.0))
            contribution = hazard / sigma_group[cen_g]
            g_eta += np.bincount(cen_g, weights=contribution, minlength=group_count)
            g_alpha += np.bincount(cen_a, weights=contribution, minlength=attraction_count)
            g_log_sigma += np.bincount(group_region[cen_g], weights=hazard * q, minlength=region_count)
        value += lambda_alpha * float(np.sum(alpha ** 2)) + lambda_beta * float(np.sum(beta[1:] ** 2)) + lambda_region * float(np.sum(regional[1:] ** 2))
        g_alpha += 2 * lambda_alpha * alpha
        g_beta = x.T @ g_eta; g_beta[1:] += 2 * lambda_beta * beta[1:]
        g_region = np.bincount(group_region, weights=g_eta, minlength=region_count); g_region[1:] += 2 * lambda_region * regional[1:]
        g_u = g_eta.copy(); g_rho = 0.0
        current = np.flatnonzero(previous >= 0)
        if len(current):
            prior = previous[current]; dynamic_residual = u[current] - rho * u[prior]
            value += lambda_dynamic * float(np.sum(dynamic_residual ** 2))
            g_u[current] += 2 * lambda_dynamic * dynamic_residual
            g_u[prior] -= 2 * lambda_dynamic * rho * dynamic_residual
            g_rho = float(np.sum(-2 * lambda_dynamic * dynamic_residual * u[prior]) * (1 - rho ** 2))
        centered_alpha_gradient = g_alpha.copy()
        for region_id in range(region_count):
            mask = attraction_region == region_id
            if mask.any(): centered_alpha_gradient[mask] -= centered_alpha_gradient[mask].mean()
        gradient = np.r_[g_beta, g_region[1:], g_u, centered_alpha_gradient, g_log_sigma, g_rho]
        return value, gradient

    def objective(vector: np.ndarray) -> float:
        return objective_gradient(vector)[0]

    def gradient(vector: np.ndarray) -> np.ndarray:
        return objective_gradient(vector)[1]

    bounds = [(None, None)] * len(initial)
    for index in range(sigma_start, rho_index): bounds[index] = (-4.0, 4.0)
    fit = minimize(objective, initial, jac=gradient, method="L-BFGS-B", bounds=bounds, options={"maxiter": 3000, "maxfun": 150000, "ftol": 1e-9})
    beta, regional, u, raw_alpha, alpha, sigma, rho, eta = unpack(fit.x)
    sampled = groups.loc[:, ["date", "region_code"]].copy()
    sampled["pressure_index"] = np.maximum(np.expm1(eta), 0.0)
    sampled["estimate_label"] = "censored_likelihood_pressure_index"
    diagnostics = pd.concat([
        pd.DataFrame({"parameter_group": "covariate", "parameter": feature_names, "estimate_standardized": beta}),
        pd.DataFrame({"parameter_group": "region_effect", "parameter": data["region_names"], "estimate_standardized": regional}),
        pd.DataFrame({"parameter_group": "dispersion", "parameter": data["region_names"], "estimate_standardized": sigma}),
        pd.DataFrame({"parameter_group": ["dynamic"], "parameter": ["rho_adjacent_sample"], "estimate_standardized": [rho]}),
    ], ignore_index=True)
    return {"converged": bool(fit.success) and bool(np.all(sigma > 0)), "objective": float(fit.fun), "message": str(fit.message), "sampled_pressure": sampled, "parameter_diagnostics": diagnostics, "group_eta": eta, "covariate_beta": beta, "region_effect": regional, "attraction_alpha": alpha, "regional_sigma": sigma, "rho": float(rho), "feature_names": list(data["feature_names"]), "feature_mean": np.asarray(data["feature_mean"], dtype=float), "feature_std": np.asarray(data["feature_std"], dtype=float), "region_names": list(data["region_names"])}


def generate_continuous_hierarchical_pressure(fit: dict[str, object], covariates: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Generate calendar-day pressure from the fitted joint covariate and region layer.

    Unsampled days use the conditional regional mean (innovation equal to zero),
    rather than a second, separately fitted ridge model.
    """
    if "date" not in covariates.columns:
        raise ValueError("covariates missing column: date")
    feature_names = list(fit["feature_names"])
    missing = set(feature_names).difference(covariates.columns)
    if missing:
        raise ValueError(f"covariates missing fitted features: {sorted(missing)}")
    cov = covariates.copy()
    cov["date"] = pd.to_datetime(cov["date"], errors="coerce")
    for index, column in enumerate(feature_names):
        cov[column] = pd.to_numeric(cov[column], errors="coerce").fillna(float(np.asarray(fit["feature_mean"])[index]))
    grid = pd.MultiIndex.from_product([cov["date"].dropna().drop_duplicates().sort_values(), regions], names=["date", "region_code"]).to_frame(index=False)
    grid = grid.merge(cov[["date", *feature_names]], on="date", how="left")
    standardized = (grid[feature_names].to_numpy(dtype=float) - np.asarray(fit["feature_mean"])) / np.asarray(fit["feature_std"])
    design = np.column_stack([np.ones(len(grid)), standardized])
    fitted_regions = {name: index for index, name in enumerate(fit["region_names"])}
    unknown = set(regions).difference(fitted_regions)
    if unknown:
        raise ValueError(f"regions not present in fitted model: {sorted(unknown)}")
    region_effect = np.asarray(fit["region_effect"], dtype=float)
    eta = design @ np.asarray(fit["covariate_beta"], dtype=float) + grid["region_code"].map(lambda value: region_effect[fitted_regions[value]]).to_numpy(dtype=float)
    grid["pressure_index"] = np.maximum(np.expm1(eta), 0.0)
    grid["estimate_label"] = "censored_likelihood_pressure_index"
    grid["projection_source"] = "joint_covariate_region_mean"
    return grid.loc[:, ["date", "region_code", "pressure_index", "estimate_label", "projection_source"]].sort_values(["date", "region_code"]).reset_index(drop=True)


def project_continuous_pressure(sampled_pressure: pd.DataFrame, covariates: pd.DataFrame, regions: list[str]) -> pd.DataFrame:
    """Project censored-likelihood sampled pressures to requested calendar days with ridge covariates."""
    source = sampled_pressure.copy(); source["date"] = pd.to_datetime(source["date"])
    cov = covariates.copy(); cov["date"] = pd.to_datetime(cov["date"])
    numeric = [c for c in cov.columns if c != "date" and pd.api.types.is_numeric_dtype(cov[c])]
    grid = pd.MultiIndex.from_product([cov["date"].drop_duplicates(), regions], names=["date","region_code"]).to_frame(index=False)
    train = source.merge(cov, on="date", how="left"); pred = grid.merge(cov, on="date", how="left")
    for c in numeric:
        fill = train[c].median(); train[c] = train[c].fillna(fill); pred[c] = pred[c].fillna(fill)
    all_regions = pd.get_dummies(pd.concat([train["region_code"], pred["region_code"]]), dtype=float)
    x = np.column_stack([np.ones(len(train)), train[numeric].to_numpy(float), all_regions.iloc[:len(train)].to_numpy(float)])
    xp = np.column_stack([np.ones(len(pred)), pred[numeric].to_numpy(float), all_regions.iloc[len(train):].to_numpy(float)])
    coef = np.linalg.pinv(x.T @ x + np.eye(x.shape[1]) * 1.0) @ x.T @ np.log1p(train["pressure_index"].to_numpy(float))
    pred["pressure_index"] = np.maximum(np.expm1(xp @ coef), 0.0); pred["estimate_label"] = "censored_likelihood_pressure_index"
    return pred.loc[:, ["date","region_code","pressure_index","estimate_label"]].sort_values(["date","region_code"]).reset_index(drop=True)


"""Convert relative regional pressure into explicitly anchor-constrained scale scenarios."""



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


"""Auditable descriptive analysis for Question 1 tourism time-series outputs."""



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

if __name__ == "__main__":
    prepare_submission_runtime(Path(__file__).resolve().parents[1])


"""Fit and export the first transparent relative-pressure baseline."""



import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]





def main() -> None:
    model_input_dir = PROJECT_ROOT / "data" / "runtime" / "model_input"
    clean_dir = PROJECT_ROOT / "data" / "runtime" / "clean"
    panel = pd.read_csv(model_input_dir / "censored_attraction_observation_panel.csv")
    calendar = pd.read_csv(clean_dir / "calendar_2015_2026.csv")
    weather = pd.read_csv(clean_dir / "daily_weather_2015_2025.csv")
    search = pd.read_csv(model_input_dir / "daily_search_theme_features_2016_2026.csv")
    calendar_dates = pd.to_datetime(calendar["date"], errors="coerce")
    weather_dates = pd.to_datetime(weather["date"], errors="coerce")
    start_date = max(calendar_dates.min(), pd.Timestamp("2023-01-01"))
    end_date = min(calendar_dates.max(), weather_dates.max(), pd.Timestamp("2025-12-31"))
    region_codes = ["BDH", "SHG", "HGA"]
    prediction_frame = pd.MultiIndex.from_product(
        [pd.date_range(start_date, end_date, freq="D"), region_codes],
        names=["date", "region_code"],
    ).to_frame(index=False)
    pressure = fit_pressure_baseline(
        panel,
        calendar,
        weather,
        search,
        prediction_frame=prediction_frame,
    )
    pressure.to_csv(
        model_input_dir / "daily_region_pressure_baseline_2023_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_scores = rolling_time_validation(
        panel,
        calendar,
        weather,
        search,
        cutoffs=["2023-06-30", "2024-06-30"],
    )
    rolling_scores.to_csv(
        model_input_dir / "pressure_baseline_rolling_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality = {
        "rows": int(len(pressure)),
        "date_start": str(pressure["date"].min().date()),
        "date_end": str(pressure["date"].max().date()),
        "region_codes": sorted(pressure["region_code"].unique().tolist()),
        "estimate_label": "relative_pressure_index",
        "rolling_validation_rows": int(len(rolling_scores)),
        "rolling_mae": rolling_scores["mae"].round(6).tolist(),
        "note": "Relative pressure only; no row is an observed or estimated absolute visitor count.",
    }
    (model_input_dir / "pressure_baseline_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


"""Fit the Q1 censored hierarchical pressure model and export daily regional pressure."""

from pathlib import Path
import sys
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]







panel = pd.read_csv(ROOT / 'data/runtime/model_input/censored_attraction_observation_panel.csv', encoding='utf-8-sig')
target_regions = ['BDH', 'HGA', 'SHG']
panel = panel.loc[panel['region_code'].isin(target_regions)].copy()
calendar = pd.read_csv(ROOT / 'data/runtime/clean/calendar_2015_2026.csv')
weather = pd.read_csv(ROOT / 'data/runtime/clean/daily_weather_2015_2025.csv')
search = pd.read_csv(ROOT / 'data/runtime/model_input/daily_search_theme_features_2016_2026.csv')
for frame in [calendar, weather, search]: frame['date'] = pd.to_datetime(frame['date'])
cov = calendar[['date','是否法定放假','是否周末']].merge(weather[['date','日均气温_℃','日降水量_mm']],on='date',how='left').merge(search[['date','theme_attraction_lag_1','theme_coastal_resort_lag_1','theme_destination_lag_1']],on='date',how='left')
cov.columns = ['date','is_holiday','is_weekend','temperature','precipitation','search_attraction_lag1','search_coastal_lag1','search_destination_lag1']
data = prepare_censored_likelihood_data(panel, cov)
fit = fit_hierarchical_censored_pressure(data)
if not fit['converged']: raise RuntimeError(fit['message'])
out = generate_continuous_hierarchical_pressure(fit, cov.loc[cov.date.between('2023-01-01','2025-12-31')], target_regions)
out.to_csv(ROOT / 'data/runtime/model_input/daily_region_censored_likelihood_pressure_2023_2025.csv', index=False, encoding='utf-8-sig')
fit['sampled_pressure'].to_csv(ROOT / 'data/runtime/model_input/censored_likelihood_sampled_pressure.csv', index=False, encoding='utf-8-sig')
fit['parameter_diagnostics'].to_csv(ROOT / 'data/runtime/model_input/censored_likelihood_parameter_diagnostics.csv', index=False, encoding='utf-8-sig')
baseline = pd.read_csv(ROOT / 'data/runtime/model_input/daily_region_pressure_baseline_2023_2025.csv', encoding='utf-8-sig')
baseline['date'] = pd.to_datetime(baseline['date'])
comparison = fit['sampled_pressure'].merge(baseline[['date','region_code','pressure_index']], on=['date','region_code'], how='left', suffixes=('_censored','_ridge'))
comparison['absolute_difference'] = (comparison['pressure_index_censored'] - comparison['pressure_index_ridge']).abs()
comparison.to_csv(ROOT / 'data/runtime/model_input/censored_likelihood_vs_ridge_validation.csv', index=False, encoding='utf-8-sig')
report={
    'continuous_rows':int(len(out)),
    'regions':sorted(out.region_code.unique().tolist()),
    'objective':float(fit['objective']),
    'sampled_groups':int(len(fit['sampled_pressure'])),
    'dynamic_rho_adjacent_sample':float(fit['rho']),
    'mean_absolute_difference_vs_ridge':float(comparison['absolute_difference'].mean()),
    'comparison_note':'same-sampled-day output difference; not a rolling forecast error',
    'unsampled_day_projection':'joint_covariate_region_mean; dynamic innovation set to zero',
}
(ROOT / 'data/runtime/model_input/censored_likelihood_quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(len(out), fit['objective'])


"""Calibrate Q1 relative pressure to visitor-scale estimates using reported anchors."""

from pathlib import Path
import sys
import pandas as pd


p=pd.read_csv(ROOT/'data/runtime/model_input/daily_region_censored_likelihood_pressure_2023_2025.csv',encoding='utf-8-sig')
a=pd.read_csv(ROOT/'data/runtime/model_input/calibration_anchor_scope_ledger.csv',encoding='utf-8-sig')
e,f=calibrate_daily_visitor_scale(p,a)
e.to_csv(ROOT/'data/runtime/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv',index=False,encoding='utf-8-sig')
f.to_csv(ROOT/'data/runtime/model_input/visitor_scale_censored_likelihood_anchor_fit.csv',index=False,encoding='utf-8-sig')
print(len(e),len(f))


"""Build the auditable descriptive-analysis outputs required by Question 1."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]





annual = pd.read_csv(ROOT / "data/runtime/clean/annual_city_tourism_1999_2024.csv", encoding="utf-8-sig")
annual = annual.loc[annual["指标"].eq("国内旅游人次")].copy()
pressure = pd.read_csv(ROOT / "data/runtime/model_input/daily_region_censored_likelihood_pressure_2023_2025.csv", encoding="utf-8-sig")
regional_visitor_estimates = pd.read_csv(ROOT / "data/runtime/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv", encoding="utf-8-sig")
parameters = pd.read_csv(ROOT / "data/runtime/model_input/censored_likelihood_parameter_diagnostics.csv", encoding="utf-8-sig")
service = pd.read_csv(ROOT / "data/runtime/clean/hourly_service_time_evidence.csv", encoding="utf-8-sig")

report = build_q1_analysis_outputs(
    annual=annual,
    pressure=pressure,
    parameters=parameters,
    service_evidence=service,
    output_directory=ROOT / "outputs/runtime/question1_analysis",
    annual_value_column="数值",
    regional_visitor_estimates=regional_visitor_estimates,
)
print(report)

if __name__ == "__main__":
    _export_submission(ROOT, "Q1_daily_visitor_estimates.csv", [
        "outputs/runtime/question1_analysis/q1_city_daily_annual_constrained.csv",
        "outputs/runtime/question1_analysis/q1_regional_pressure_decomposition.csv",
    ])
    _export_submission(ROOT, "Q1_validation_summary.csv", [
        "outputs/runtime/question1_analysis/q1_factor_strength.csv",
        "outputs/runtime/question1_analysis/q1_regional_season_classification.csv",
    ])
