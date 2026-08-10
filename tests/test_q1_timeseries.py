import pandas as pd
import json
import pytest

from src.analysis.q1_timeseries import (
    analyze_city_annual_trend,
    classify_regional_seasons,
    decompose_regional_pressure,
    rank_factor_strength,
    build_service_time_scenarios,
    build_q1_analysis_outputs,
    construct_annual_constrained_city_daily_series,
    build_entry_exit_time_profiles,
)


def test_city_annual_trend_residual_reconstructs_official_total():
    source = pd.DataFrame({"year": [2020, 2021, 2022], "tourists": [10.0, 16.0, 22.0]})

    result = analyze_city_annual_trend(source, value_column="tourists")

    assert (result["official_tourists"] == result["trend_component"] + result["random_component"]).all()


def test_regional_decomposition_components_reconstruct_log_pressure():
    source = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=90, freq="D"),
        "region_code": ["BDH"] * 90,
        "pressure_index": [1.0 + (day % 12) for day in range(90)],
    })

    result = decompose_regional_pressure(source, rolling_window=15)

    reconstructed = result["trend_component"] + result["seasonal_component"] + result["random_component"]
    assert (result["log_pressure"] - reconstructed).abs().max() < 1e-10


def test_season_classification_covers_every_region_month_once():
    source = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=366, freq="D").tolist() * 2,
        "region_code": ["BDH"] * 366 + ["HGA"] * 366,
        "pressure_index": list(range(366)) + list(reversed(range(366))),
    })

    result = classify_regional_seasons(source)

    assert len(result) == 24
    assert set(result["season_label"]).issubset({"peak_season", "shoulder_season", "off_season"})


def test_zero_pressure_tied_at_lower_quantile_is_still_classified_as_off_season():
    source = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=365, freq="D"),
        "region_code": ["SHG"] * 365,
        "pressure_index": [0.0] * 151 + [2.0] * 214,
    })

    result = classify_regional_seasons(source)

    assert (result.loc[result["monthly_median_pressure"].eq(0), "season_label"] == "off_season").all()


def test_factor_strength_ranks_covariates_by_absolute_standardized_effect():
    source = pd.DataFrame({
        "parameter_group": ["covariate", "covariate", "covariate", "dynamic"],
        "parameter": ["intercept", "rain", "holiday", "rho_adjacent_sample"],
        "estimate_standardized": [1.0, -0.2, 0.7, 0.5],
    })

    result = rank_factor_strength(source)

    assert result["parameter"].tolist() == ["holiday", "rain"]
    assert result["effect_direction"].tolist() == ["positive", "negative"]


def test_service_time_scenarios_do_not_create_hourly_visitor_counts():
    source = pd.DataFrame({
        "region_name": ["Beidaihe_District"],
        "period_start": ["2024-07-01"],
        "period_end": ["2024-08-31"],
        "day_type": ["summer_operating_day"],
        "time_start": ["07:30"],
        "time_end": ["19:30"],
        "metric_name": ["service window"],
        "evidence_type": ["service_time_or_qualitative_peak_not_hourly_count"],
        "data_status": ["verified_schedule_not_entry_distribution"],
        "source_url": ["https://example.test"],
        "is_hourly_count": [0],
    })

    result = build_service_time_scenarios(source)

    assert result.loc[0, "not_observed_hourly_flow"]
    assert pd.isna(result.loc[0, "hourly_visitor_count"])


def test_q1_output_builder_writes_all_auditable_analysis_artifacts(tmp_path):
    annual = pd.DataFrame({"year": [2023, 2024], "tourists": [80.0, 90.0]})
    pressure = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=32, freq="D"), "region_code": ["BDH"] * 32, "pressure_index": range(32)})
    parameters = pd.DataFrame({"parameter_group": ["covariate"], "parameter": ["holiday"], "estimate_standardized": [0.2]})
    service = pd.DataFrame({"region_name": ["BDH"], "period_start": ["2024-07-01"], "period_end": ["2024-08-31"], "day_type": ["summer"], "time_start": ["08:00"], "time_end": ["18:00"], "metric_name": ["service"], "evidence_type": ["proxy"], "data_status": ["verified"], "source_url": ["https://example.test"], "is_hourly_count": [0]})

    regional_estimates = pressure.assign(visitor_estimate_baseline=pressure["pressure_index"] + 1.0)
    report = build_q1_analysis_outputs(annual, pressure, parameters, service, tmp_path, annual_value_column="tourists", regional_visitor_estimates=regional_estimates)

    assert report["regional_pressure_rows"] == 32
    assert (tmp_path / "q1_city_annual_trend.csv").exists()
    assert (tmp_path / "q1_regional_pressure_decomposition.csv").exists()
    assert (tmp_path / "q1_city_daily_annual_constrained.csv").exists()
    assert (tmp_path / "q1_city_entry_exit_time_profiles.csv").exists()
    assert json.loads((tmp_path / "q1_analysis_quality_report.json").read_text(encoding="utf-8"))["hourly_flow_claim"] == "not_observed"


def test_city_daily_series_sums_to_official_annual_total_after_anchor_constraint():
    regional = pd.DataFrame({
        "date": ["2024-07-01", "2024-07-01", "2024-07-02", "2024-07-02"],
        "region_code": ["BDH", "HGA", "BDH", "HGA"],
        "visitor_estimate_baseline": [2.0, 3.0, 1.0, 4.0],
    })
    annual = pd.DataFrame({"year": [2024], "official_tourists": [100.0]})

    result = construct_annual_constrained_city_daily_series(regional, annual, annual_value_column="official_tourists")

    assert result["city_daily_visitor_estimate"].sum() == 100000.0
    assert result["estimate_label"].eq("annual_anchor_constrained_city_daily_estimate").all()


def test_entry_exit_profiles_are_normalized_scenarios_not_hourly_observations():
    evidence = pd.DataFrame({
        "region_name": ["Beidaihe_District"], "period_start": ["2024-07-01"], "period_end": ["2024-08-31"],
        "day_type": ["summer"], "time_start": ["08:00"], "time_end": ["18:00"],
        "metric_name": ["day service"], "evidence_type": ["service_time"], "data_status": ["verified"],
        "source_url": ["https://example.test"], "is_hourly_count": [0],
    })

    result = build_entry_exit_time_profiles(evidence, bin_minutes=30)

    assert result["entry_weight"].sum() == pytest.approx(1.0)
    assert result["exit_weight"].sum() == pytest.approx(1.0)
    assert result["profile_status"].eq("assumption_driven_not_observed_hourly_flow").all()
