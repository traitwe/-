"""Build concise delivery tables from the reproducible runtime data and outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(root: Path, relative_path: str) -> pd.DataFrame:
    """Read one UTF-8 CSV and retain its origin for traceability after merging."""
    frame = pd.read_csv(root / relative_path, encoding="utf-8-sig")
    frame.insert(0, "source_table", Path(relative_path).name)
    return frame


def _concat(root: Path, relative_paths: list[str]) -> pd.DataFrame:
    return pd.concat([_read(root, path) for path in relative_paths], ignore_index=True, sort=False)


def _write(frame: pd.DataFrame, directory: Path, filename: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def build_submission_delivery(
    root: Path = PROJECT_ROOT,
    data_directory: Path | None = None,
    result_directory: Path | None = None,
) -> dict[str, list[Path]]:
    """Create eight canonical data tables and nine canonical result tables."""
    root = root.resolve()
    data_dir = data_directory or root / "data/submission"
    result_dir = result_directory or root / "outputs/submission"

    calendar = _read(root, "data/runtime/clean/calendar_2015_2026.csv")
    weather = _read(root, "data/runtime/clean/daily_weather_2015_2025.csv").drop(columns="source_table")
    search = _read(root, "data/runtime/model_input/daily_search_theme_features_2016_2026.csv").drop(columns="source_table")
    for frame in (calendar, weather, search):
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    drivers = calendar.drop(columns="source_table").merge(weather, on="date", how="left", suffixes=("", "_weather"))
    drivers = drivers.merge(search, on="date", how="left", suffixes=("", "_search"))

    data_artifacts = [
        _write(_concat(root, [
            "data/runtime/clean/annual_city_tourism_1999_2024.csv",
            "data/runtime/clean/raw_cleaned/qinhuangdao_city_tourism_annual_revised.csv",
            "data/runtime/clean/raw_cleaned/qinhuangdao_public_visitor_flow_anchors_2023_2025.csv",
            "data/runtime/clean/raw_cleaned/qinhuangdao_summer_regional_flow_anchors_2023_2025.csv",
            "data/runtime/clean/raw_cleaned/aranya_beidaihe_new_district_summer_flow_anchors_2023_2025.csv",
            "data/runtime/clean/raw_cleaned/beidaihe_tourism_anchors_revised.csv",
            "data/runtime/clean/raw_cleaned/haigang_aranya_tourism_anchors.csv",
            "data/runtime/clean/raw_cleaned/shanhaiguan_tourism_anchors.csv",
        ]), data_dir, "01_tourism_flow_anchor_register.csv"),
        _write(drivers, data_dir, "02_daily_driver_panel_2015_2026.csv"),
        _write(_read(root, "data/runtime/model_input/censored_attraction_observation_panel.csv"), data_dir, "03_attraction_observation_panel.csv"),
        _write(_concat(root, [
            "data/runtime/clean/poi_attraction.csv", "data/runtime/clean/poi_hotel.csv",
            "data/runtime/clean/poi_parking.csv", "data/runtime/clean/poi_transit.csv",
        ]), data_dir, "04_poi_facility_inventory.csv"),
        _write(_concat(root, [
            "data/runtime/clean/attraction_effective_area.csv", "data/runtime/clean/capacity_space_standard.csv",
            "data/runtime/clean/raw_cleaned/q3_resource_constraints_researched.csv",
            "data/runtime/clean/raw_cleaned/q3_parking_shuttle_real_anchor_register.csv",
            "data/runtime/clean/raw_cleaned/tourism_resource_capacity_anchors.csv",
            "data/runtime/clean/hourly_service_time_evidence.csv",
        ]), data_dir, "05_resource_constraint_register.csv"),
        _write(_read(root, "data/runtime/clean/raw_cleaned/qinhuangdao_hotel_room_ledger_2026-08-10.csv"), data_dir, "06_hotel_room_ledger.csv"),
        _write(_concat(root, [
            "data/runtime/clean/reservation_ratio_scenarios.csv", "data/runtime/model_input/q3_absolute_planning_scenarios.csv",
            "data/runtime/model_input/q3_vot_scenarios.csv",
        ]), data_dir, "07_model_parameter_scenarios.csv"),
        _write(_concat(root, [
            "data/runtime/clean/raw_inventory.csv", "data/runtime/clean/raw_cleaned/supplementary_data_collection_register.csv",
            "data/runtime/model_input/calibration_anchor_scope_ledger.csv",
        ]), data_dir, "08_data_source_quality_register.csv"),
    ]

    result_artifacts = [
        _write(_concat(root, [
            "outputs/runtime/question1_analysis/q1_city_daily_annual_constrained.csv",
            "outputs/runtime/question1_analysis/q1_regional_pressure_decomposition.csv",
        ]), result_dir, "Q1_daily_visitor_estimates.csv"),
        _write(_concat(root, [
            "outputs/runtime/question1_analysis/q1_factor_strength.csv",
            "outputs/runtime/question1_analysis/q1_regional_season_classification.csv",
        ]), result_dir, "Q1_validation_summary.csv"),
        _write(_concat(root, [
            "outputs/runtime/question2_analysis/q2_city_monthly_forecast_2026.csv",
            "outputs/runtime/question2_analysis/q2_region_daily_forecast_2026.csv",
        ]), result_dir, "Q2_2026_forecast.csv"),
        _write(_concat(root, [
            "outputs/runtime/question2_analysis/q2_annual_model_comparison.csv",
            "outputs/runtime/question2_analysis/q2_daily_model_comparison.csv",
            "outputs/runtime/question2_analysis/q2_daily_rolling_origin_validation.csv",
            "outputs/runtime/question2_analysis/q2_daily_conformal_calibration_2026.csv",
            "outputs/runtime/question2_analysis/q2_external_anchor_validation.csv",
        ]), result_dir, "Q2_validation_summary.csv"),
        _write(_read(root, "outputs/runtime/question3_analysis/q3_absolute_daily_optimized_plan_2026.csv"), result_dir, "Q3_daily_resource_plan.csv"),
        _write(_concat(root, [
            "outputs/runtime/question3_analysis/q3_absolute_seasonal_resource_plan_2026.csv",
            "outputs/runtime/question3_analysis/q3_beidaihe_absolute_summer_temporary_plan_2026.csv",
        ]), result_dir, "Q3_seasonal_recommendations.csv"),
        _write(_concat(root, [
            "outputs/runtime/question3_analysis/q3_absolute_resource_sensitivity_2026.csv",
            "outputs/runtime/question3_analysis/q3_absolute_cost_experience_pareto_2026.csv",
            "outputs/runtime/question3_analysis/q3_bdh_temporary_capacity_relaxation.csv",
        ]), result_dir, "Q3_sensitivity_summary.csv"),
        _write(_concat(root, [
            "outputs/runtime/question4_analysis/q4_event_counterfactual_2026.csv",
            "outputs/runtime/question4_analysis/q4_rain_counterfactual_2026.csv",
        ]), result_dir, "Q4_scenario_reoptimization.csv"),
        _write(_concat(root, [
            "outputs/runtime/question4_analysis/q4_baseline_reoptimization_summary_2026.csv",
            "outputs/runtime/question4_analysis/q4_robustness_sensitivity_ranking.csv",
            "outputs/runtime/question4_analysis/q4_scenario_window_adjustment_summary.csv",
            "outputs/runtime/question4_analysis/q4_lambda_tradeoff_2026.csv",
        ]), result_dir, "Q4_robustness_summary.csv"),
    ]
    return {"data": data_artifacts, "results": result_artifacts}


if __name__ == "__main__":
    generated = build_submission_delivery()
    for category, paths in generated.items():
        print(f"{category}: {len(paths)} files")

