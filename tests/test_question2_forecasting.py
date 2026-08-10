import numpy as np
import pandas as pd
import pytest

from src.models.question2_forecasting import (
    DailyRidgeForecaster,
    allocate_monthly_total,
    annual_rolling_origin_splits,
    build_daily_calendar_features,
    fit_daily_ridge_forecaster,
    fit_annual_log_trend,
    forecast_annual_total,
    forecast_annual_candidate,
    evaluate_annual_candidates,
    select_annual_candidate,
    predict_daily_ridge_forecaster,
    scenario_adjust_weather,
)
from src.features.search_features import build_search_theme_features
from src.analysis.q2_forecast_outputs import _attach_covariates, _daily_comparison, apply_regional_relative_split_conformal_interval, apply_relative_split_conformal_interval, build_climatological_covariates, build_external_anchor_validation, build_q2_diagnostic_figures, build_question2_outputs, select_regional_search_lags


def test_annual_rolling_origin_splits_keep_all_training_years_before_the_target():
    result = annual_rolling_origin_splits([2019, 2020, 2021, 2022, 2023], min_train_years=3)

    assert result == [(2019, 2021, 2022), (2019, 2022, 2023)]


def test_annual_forecast_has_a_nonnegative_ordered_interval():
    annual = pd.DataFrame({"year": [2019, 2020, 2021, 2022, 2023], "total": [100.0, 105.0, 112.0, 120.0, 132.0]})

    model = fit_annual_log_trend(annual, value_column="total")
    forecast = forecast_annual_total(model, target_year=2024)

    assert forecast["lower"] >= 0
    assert forecast["lower"] <= forecast["point"] <= forecast["upper"]
    assert forecast["point"] > 132.0


def test_monthly_allocation_preserves_annual_total_and_month_labels():
    result = allocate_monthly_total(1200.0, np.arange(1, 13, dtype=float), year=2026)

    assert result["month"].tolist() == list(range(1, 13))
    assert result["visitor_estimate"].sum() == 1200.0
    assert (result["visitor_estimate"] >= 0).all()


def test_annual_candidates_are_compared_only_on_forward_rolling_targets():
    annual = pd.DataFrame({"year": range(2015, 2024), "total": [100, 102, 105, 111, 115, 117, 123, 129, 134]})

    result = evaluate_annual_candidates(annual, value_column="total", min_train_years=4)

    assert set(result["candidate"]) == {"weighted_log_trend", "last_year_naive", "recent_median_growth"}
    assert result["test_rows"].eq(5).all()
    assert (result[["mae", "rmse", "smape"]] >= 0).all().all()


def test_annual_candidate_selection_prefers_lowest_smape_then_simpler_model():
    scores = pd.DataFrame({
        "candidate": ["weighted_log_trend", "last_year_naive", "recent_median_growth"],
        "mae": [8.0, 3.0, 4.0], "rmse": [9.0, 4.0, 5.0], "smape": [5.0, 5.0, 6.0],
    })

    assert select_annual_candidate(scores) == "last_year_naive"


def test_selected_annual_candidate_forecast_has_its_own_interval():
    annual = pd.DataFrame({"year": range(2015, 2024), "total": [100, 102, 105, 111, 115, 117, 123, 129, 134]})

    forecast = forecast_annual_candidate(annual, "total", 2024, "recent_median_growth", min_train_years=4)

    assert forecast["candidate"] == "recent_median_growth"
    assert forecast["lower"] <= forecast["point"] <= forecast["upper"]


def test_daily_calendar_features_are_deterministic_from_dates():
    result = build_daily_calendar_features(pd.DataFrame({"date": ["2026-07-04", "2026-07-06"]}))

    assert result["is_summer"].tolist() == [1.0, 1.0]
    assert result["is_weekend"].tolist() == [1.0, 0.0]
    assert {"annual_sin", "annual_cos"}.issubset(result.columns)
    assert result["weekday_5"].tolist() == [1.0, 0.0]


def test_search_theme_features_include_supported_short_lags_for_regional_selection():
    source = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=8, freq="D"),
        "keyword": ["秦皇岛旅游"] * 8,
        "search_index": np.arange(8, dtype=float),
        "quality_grade": ["A"] * 8,
        "theme": ["destination"] * 8,
    })

    result = build_search_theme_features(source)

    assert {"theme_destination_lag_1", "theme_destination_lag_2", "theme_destination_lag_3", "theme_destination_lag_7"}.issubset(result.columns)


def test_daily_forecaster_uses_a_positive_training_floor_for_nonnegative_series():
    source = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=14), "region_code": ["SHG"] * 14, "target": [2.0] * 14})
    model = fit_daily_ridge_forecaster(source, target_column="target", dynamic=False)
    future = pd.DataFrame({"date": ["2026-01-05"], "region_code": ["SHG"]})
    assert predict_daily_ridge_forecaster(model, future)["prediction_q50"].iloc[0] > 0


def test_prediction_floor_does_not_collapse_the_entire_interval_to_one_value():
    model = DailyRidgeForecaster(
        feature_names=("intercept", "is_summer", "annual_sin", "annual_cos", "weekday_1", "weekday_2", "weekday_3", "weekday_4", "weekday_5", "weekday_6"),
        feature_means=(0.0,) * 10,
        feature_scales=(1.0,) * 10,
        coefficients=(-0.7,) + (0.0,) * 9,
        residual_std=1.5,
        region_levels=("SHG",),
        dynamic=False,
        prediction_floor=2.0,
    )

    prediction = predict_daily_ridge_forecaster(model, pd.DataFrame({"date": ["2026-01-05"], "region_code": ["SHG"]}))

    assert prediction.loc[0, "prediction_q10"] < prediction.loc[0, "prediction_q90"]
    assert prediction.loc[0, "prediction_q10"] <= prediction.loc[0, "prediction_q50"] <= prediction.loc[0, "prediction_q90"]


def test_daily_ridge_forecast_orders_its_quantiles():
    source = pd.DataFrame({
        "date": pd.date_range("2025-07-01", periods=12, freq="D"),
        "region_code": ["BDH"] * 12,
        "visitor_estimate_baseline": np.arange(10.0, 22.0),
        "rain_mm": [0.0, 1.0] * 6,
        "search_lag_1": np.arange(12.0),
    })

    model = fit_daily_ridge_forecaster(source, target_column="visitor_estimate_baseline", dynamic=True, ridge_alpha=1.0)
    prediction = predict_daily_ridge_forecaster(model, source)

    assert (prediction["prediction_q10"] <= prediction["prediction_q50"]).all()
    assert (prediction["prediction_q50"] <= prediction["prediction_q90"]).all()
    assert prediction["estimate_label"].eq("anchor_constrained_daily_visitor_forecast").all()


def test_calendar_baseline_excludes_weather_and_search_features():
    source = pd.DataFrame({"date": pd.date_range("2025-07-01", periods=4, freq="D"), "region_code": ["BDH"] * 4, "target": [2.0, 3.0, 4.0, 5.0], "rain_mm": [1.0, 2.0, 3.0, 4.0], "search_lag_1": [5.0, 6.0, 7.0, 8.0]})

    model = fit_daily_ridge_forecaster(source, target_column="target", dynamic=False)

    assert "rain_mm" not in model.feature_names
    assert "search_lag_1" not in model.feature_names


def test_weather_scenario_only_changes_weather_input_columns():
    source = pd.DataFrame({"date": ["2026-07-01"], "rain_mm": [4.0], "temperature_c": [28.0], "search_lag_1": [10.0]})

    result = scenario_adjust_weather(source, rain_multiplier=2.0, temperature_shift_c=1.5)

    assert result.loc[0, "rain_mm"] == 8.0
    assert result.loc[0, "temperature_c"] == 29.5
    assert result.loc[0, "search_lag_1"] == 10.0


def test_weather_scenario_supports_additive_rainfall_shock():
    source = pd.DataFrame({"date": ["2026-07-01"], "rain_mm": [0.0]})
    result = scenario_adjust_weather(source, rain_add_mm=20.0)
    assert result.loc[0, "rain_mm"] == 20.0


def test_relative_split_conformal_interval_expands_existing_quantiles_by_calibrated_relative_error():
    prediction = pd.DataFrame({"prediction_q10": [9.0], "prediction_q50": [10.0], "prediction_q90": [11.0]})

    adjusted, radius = apply_relative_split_conformal_interval(
        prediction,
        calibration_actual=pd.Series([5.0, 15.0]),
        calibration_point=pd.Series([10.0, 10.0]),
        coverage=0.8,
    )

    assert radius == pytest.approx(0.5)
    assert adjusted.loc[0, "prediction_q10"] == pytest.approx(5.0)
    assert adjusted.loc[0, "prediction_q90"] == pytest.approx(15.0)


def test_regional_conformal_calibration_keeps_distinct_region_radii():
    prediction = pd.DataFrame({
        "region_code": ["BDH", "HGA"],
        "prediction_q10": [9.0, 9.0], "prediction_q50": [10.0, 10.0], "prediction_q90": [11.0, 11.0],
    })
    calibration = pd.DataFrame({
        "region_code": ["BDH", "BDH", "HGA", "HGA"],
        "actual": [5.0, 15.0, 9.0, 11.0], "point": [10.0, 10.0, 10.0, 10.0],
    })

    adjusted, radii = apply_regional_relative_split_conformal_interval(
        prediction, calibration, actual_column="actual", point_column="point", coverage=0.8, min_region_rows=2,
    )

    assert radii.set_index("region_code").loc["BDH", "relative_radius"] == pytest.approx(0.5)
    assert radii.set_index("region_code").loc["HGA", "relative_radius"] == pytest.approx(0.1)
    assert adjusted.loc[adjusted["region_code"].eq("BDH"), "prediction_q10"].iloc[0] == pytest.approx(5.0)
    assert adjusted.loc[adjusted["region_code"].eq("HGA"), "prediction_q10"].iloc[0] == pytest.approx(9.0)


def test_daily_comparison_uses_latest_year_with_sufficient_observations_and_reports_interval_metrics():
    dates = pd.date_range("2023-01-01", "2025-12-31", freq="D")
    regional = pd.DataFrame({"date": dates, "region_code": "SHG", "pressure_index": np.linspace(1.0, 2.0, len(dates)), "baseline_scale": 100.0})
    observed_dates = (
        list(pd.date_range("2023-01-01", periods=60, freq="D"))
        + list(pd.date_range("2024-01-01", periods=60, freq="D"))
        + list(pd.date_range("2025-01-01", periods=14, freq="D"))
    )
    observed = pd.DataFrame({"date": observed_dates, "region_code": "SHG", "visitor_index": [0.5] * len(observed_dates)})
    result, unavailable = _daily_comparison(regional, observed)
    assert not unavailable
    assert result["holdout_year"].eq(2024).all()
    assert "historical_weekday_median" in set(result["candidate"])
    assert {"picp_80", "mean_interval_width", "interval_calibration", "conformal_relative_radius"}.issubset(result.columns)
    assert result.loc[result["candidate"].eq("dynamic_ridge"), "interval_calibration"].iloc[0] == "regional_relative_split_conformal_pre_2024"


def test_climatological_covariates_use_only_history_before_forecast_year():
    covariates = pd.DataFrame({
        "date": ["2024-07-01", "2025-07-01", "2026-07-01"],
        "rain_mm": [2.0, 4.0, 999.0], "temperature_c": [25.0, 27.0, 999.0], "search_lag_1": [10.0, 14.0, 999.0],
    })

    result = build_climatological_covariates(covariates, 2026)
    july_first = result.loc[result["date"].eq(pd.Timestamp("2026-07-01"))].iloc[0]

    assert july_first["rain_mm"] == 3.0
    assert july_first["temperature_c"] == 26.0
    assert july_first["search_lag_1"] == 12.0


def test_regional_search_lag_selection_uses_only_observed_training_dates_and_shorter_lag_ties():
    dates = pd.date_range("2023-01-01", periods=24, freq="D")
    search = pd.DataFrame({
        "date": dates,
        "theme_destination_lag_1": np.arange(24, dtype=float),
        "theme_destination_lag_2": np.arange(24, dtype=float)[::-1],
        "theme_destination_lag_3": np.repeat([1.0, 2.0], 12),
        "theme_destination_lag_7": np.ones(24),
    })
    observed = pd.DataFrame({
        "date": list(dates) * 2,
        "region_code": ["BDH"] * 24 + ["HGA"] * 24,
        "visitor_index": list(np.arange(24, dtype=float)) + list(np.arange(24, dtype=float)[::-1]),
    })

    result = select_regional_search_lags(observed, search, min_rows=20)

    selected = result.set_index("region_code")["selected_search_column"].to_dict()
    assert selected == {"BDH": "theme_destination_lag_1", "HGA": "theme_destination_lag_1"}
    assert result["selection_status"].eq("selected_from_observed_training_dates").all()


def test_attach_covariates_respects_region_specific_search_lag_values():
    target = pd.DataFrame({"date": ["2026-07-01", "2026-07-01"], "region_code": ["BDH", "HGA"]})
    covariates = pd.DataFrame({"date": ["2026-07-01", "2026-07-01"], "region_code": ["BDH", "HGA"], "search_lag_1": [10.0, 20.0]})

    result = _attach_covariates(target, covariates)

    assert result["search_lag_1"].tolist() == [10.0, 20.0]


def test_external_anchor_validation_calculates_error_only_for_comparable_citywide_tourism_scope():
    anchors = pd.DataFrame({"period_start": ["2024-05-01", "2024-05-01"], "period_end": ["2024-05-02", "2024-05-02"], "region_name": ["Qinhuangdao_City", "Qinhuangdao_City"], "visitor_count_10k_persons": [2.0, 2.0], "value_qualifier": ["exact", "exact"], "metric_scope": ["citywide_tourist_visitors", "citywide_scenic_scope"], "source_url": ["a", "b"]})
    city = pd.DataFrame({"date": ["2024-05-01", "2024-05-02"], "city_daily_visitor_estimate": [11000.0, 10000.0]})
    regional = pd.DataFrame({"date": ["2024-05-01", "2024-05-02"], "region_code": ["BDH", "BDH"], "visitor_estimate_baseline": [5000.0, 5000.0]})

    result = build_external_anchor_validation(anchors, city, regional)

    assert result.loc[0, "comparison_status"] == "directly_comparable"
    assert result.loc[0, "relative_error"] == pytest.approx(0.05)
    assert result.loc[1, "comparison_status"] == "scope_not_directly_comparable"
    assert pd.isna(result.loc[1, "relative_error"])


def test_q2_output_builder_writes_constrained_city_and_three_region_forecasts(tmp_path):
    annual = pd.DataFrame({"year": [2019, 2020, 2021, 2022, 2023, 2024], "official_total": [100.0, 80.0, 90.0, 95.0, 110.0, 120.0]})
    dates = pd.date_range("2023-01-01", "2025-01-25", freq="D")
    regional = pd.DataFrame({
        "date": np.repeat(dates, 3),
        "region_code": ["BDH", "HGA", "SHG"] * len(dates),
        "pressure_index": np.tile(np.linspace(2.0, 8.0, len(dates)), 3),
        "baseline_scale": 100.0,
    })
    observed = regional.loc[regional["date"].dt.year.isin([2023, 2025]), ["date", "region_code", "pressure_index"]].rename(columns={"pressure_index": "visitor_index"})
    observed = pd.concat([observed, pd.DataFrame({"date": ["2021-01-01"], "region_code": ["OUTSIDE"], "visitor_index": [99.0]})], ignore_index=True)
    regional["date"] = regional["date"].dt.strftime("%Y-%m-%d")
    covariates = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "rain_mm": 1.0, "temperature_c": 25.0, "search_lag_1": 10.0})

    report = build_question2_outputs(annual, regional, observed, tmp_path, forecast_year=2026, daily_covariates=covariates)

    monthly = pd.read_csv(tmp_path / "q2_city_monthly_forecast_2026.csv", encoding="utf-8-sig")
    daily = pd.read_csv(tmp_path / "q2_region_daily_forecast_2026.csv", encoding="utf-8-sig")
    assert report["forecast_year"] == 2026
    assert monthly["visitor_estimate"].sum() == pytest.approx(report["annual_forecast_point"])
    assert set(daily["region_code"]) == {"BDH", "HGA", "SHG"}
    assert pd.to_datetime(daily["date"]).dt.year.eq(2026).all()
    assert daily["estimate_label"].eq("anchor_constrained_daily_visitor_forecast").all()
    assert daily["interval_calibration"].eq("regional_relative_split_conformal_pre_2025").all()
    assert report["daily_validation_scope"] == "raw_attraction_observation_dates_only"
    assert (tmp_path / "q2_annual_model_comparison.csv").exists()
    assert report["annual_selected_model"] in {"weighted_log_trend", "last_year_naive", "recent_median_growth"}
    assert report["daily_validation_covariate_mode"] == "conditional_on_realized_weather_and_lagged_search"
    assert report["weather_response_model"] in {"dynamic_ridge_sensitivity_only", "selected_daily_model"}
    assert 0.0 <= report["rain_shock_effective_share"] <= 1.0
    calibration = pd.read_csv(tmp_path / "q2_daily_conformal_calibration_2026.csv", encoding="utf-8-sig")
    assert set(calibration["calibration_status"]).issubset({"region_specific", "global_fallback_insufficient_region_calibration"})
    assert set(daily["region_code"]).issubset(set(calibration["region_code"]))


def test_q2_diagnostic_figures_are_created_from_forecast_tables(tmp_path):
    monthly = pd.DataFrame({"month": [1, 2], "visitor_estimate": [100.0, 120.0], "annual_forecast_lower": [1000.0, 1000.0], "annual_forecast_upper": [1400.0, 1400.0]})
    daily = pd.DataFrame({"date": ["2026-07-01", "2026-07-01"], "region_code": ["BDH", "HGA"], "visitor_estimate_q10": [10.0, 20.0], "visitor_estimate_q50": [15.0, 25.0], "visitor_estimate_q90": [20.0, 30.0]})

    build_q2_diagnostic_figures(monthly, daily, tmp_path)

    assert (tmp_path / "q2_city_monthly_forecast.png").exists()
    assert (tmp_path / "q2_summer_regional_daily_forecast.png").exists()
