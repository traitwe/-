"""Run the Question 2 two-scale forecast pipeline."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.q2_forecast_outputs import build_external_anchor_validation, build_question2_outputs


annual = pd.read_csv(ROOT / "data/clean/annual_city_tourism_1999_2024.csv", encoding="utf-8-sig")
annual = annual.loc[annual["\u6307\u6807"].eq("\u56fd\u5185\u65c5\u6e38\u4eba\u6b21"), ["year", "\u6570\u503c", "\u5355\u4f4d"]].rename(columns={"\u6570\u503c": "official_total"})
if not annual["\u5355\u4f4d"].eq("\u5343\u4eba\u6b21").all():
    raise ValueError("official annual unit must be thousand visitor-trips before conversion")
annual["official_total"] = annual["official_total"] * 1000.0
regional = pd.read_csv(ROOT / "data/model_input/daily_region_visitor_scale_censored_likelihood_2023_2025.csv", encoding="utf-8-sig")
observed = pd.read_csv(ROOT / "data/model_input/censored_attraction_observation_panel.csv", encoding="utf-8-sig")
weather = pd.read_csv(ROOT / "data/clean/daily_weather_2015_2025.csv", encoding="utf-8-sig")
weather = weather.loc[:, ["date", "\u65e5\u5747\u6c14\u6e29_\u2103", "\u65e5\u964d\u6c34\u91cf_mm"]].rename(columns={"\u65e5\u5747\u6c14\u6e29_\u2103": "temperature_c", "\u65e5\u964d\u6c34\u91cf_mm": "rain_mm"})
search = pd.read_csv(ROOT / "data/model_input/daily_search_theme_features_2016_2026.csv", encoding="utf-8-sig")
search = search.loc[:, ["date", "theme_destination_lag_1"]].rename(columns={"theme_destination_lag_1": "search_lag_1"})
covariates = weather.merge(search, on="date", how="outer")
output = ROOT / "outputs/question2_analysis"
print(build_question2_outputs(annual, regional, observed, output, daily_covariates=covariates))
anchors = pd.read_csv(ROOT / "data/clean/raw_cleaned/qinhuangdao_summer_regional_flow_anchors_2023_2025.csv", encoding="utf-8-sig")
city_daily = pd.read_csv(ROOT / "outputs/question1_analysis/q1_city_daily_annual_constrained.csv", encoding="utf-8-sig")
validation = build_external_anchor_validation(anchors, city_daily, regional)
validation.to_csv(output / "q2_external_anchor_validation.csv", index=False, encoding="utf-8-sig")
