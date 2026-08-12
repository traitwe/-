"""Fit the Q1 censored hierarchical pressure model and export daily regional pressure."""

from pathlib import Path
import sys
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.hierarchical_censored_pressure import (
    prepare_censored_likelihood_data,
    fit_hierarchical_censored_pressure,
    generate_continuous_hierarchical_pressure,
)

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
