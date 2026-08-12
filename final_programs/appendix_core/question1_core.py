# 问题一：状态划分、左删失似然与游客规模标定
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

def censored_negative_log_posterior(beta, x, observed, threshold, censored):
    """排名上榜值按正态密度计，未上榜值按左删失概率计。"""
    mean = x @ beta[:-1]
    sigma = max(beta[-1], 1e-6)
    observed_part = -np.log(np.maximum(
        np.exp(-0.5 * ((observed - mean) / sigma) ** 2) / sigma, 1e-12)).sum()
    censored_part = -log_ndtr((threshold - mean[censored]) / sigma).sum()
    return observed_part + censored_part + 0.1 * np.square(beta[:-1]).sum()

def estimate_pressure(x, observed, threshold, censored):
    initial = np.r_[np.zeros(x.shape[1]), 1.0]
    result = minimize(censored_negative_log_posterior, initial,
                      args=(x, observed, threshold, censored), method="L-BFGS-B")
    return x @ result.x[:-1]

def calibrate_visitors(pressure, anchor_pressure, anchor_visitors):
    scale = np.median(anchor_visitors / np.maximum(anchor_pressure, 1e-8))
    return np.maximum(pressure * scale, 0.0)
