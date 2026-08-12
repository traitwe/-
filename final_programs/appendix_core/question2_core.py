# 问题二：年总量—月度份额—日尺度协变量预测
import numpy as np

def fit_ridge(x, y, penalty=1.0):
    x1 = np.c_[np.ones(len(x)), x]
    return np.linalg.solve(x1.T @ x1 + penalty * np.eye(x1.shape[1]), x1.T @ y)

def forecast_daily(train_x, train_y, target_x, annual_total, month_share):
    beta = fit_ridge(train_x, np.log1p(train_y))
    raw = np.expm1(np.c_[np.ones(len(target_x)), target_x] @ beta).clip(0)
    raw = raw * np.asarray(month_share)
    # 在每月内保持协变量造成的相对起伏，并与年度总量一致。
    return annual_total * raw / max(raw.sum(), 1e-8)

def split_conformal_interval(prediction, residuals, level=0.90):
    radius = np.quantile(np.abs(residuals), level)
    return prediction - radius, prediction + radius
