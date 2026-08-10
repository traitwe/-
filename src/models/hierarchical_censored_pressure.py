"""Data preparation for the Question 1 hierarchical left-censored pressure model."""

from __future__ import annotations

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
