from pathlib import Path


def test_submission_delivery_builds_eight_data_and_nine_result_tables(tmp_path):
    from scripts.build_submission_delivery import build_submission_delivery

    root = Path(__file__).resolve().parents[1]
    artifacts = build_submission_delivery(root, data_directory=tmp_path / "data", result_directory=tmp_path / "results")

    data_files = sorted(path.name for path in artifacts["data"])
    result_files = sorted(path.name for path in artifacts["results"])
    assert len(data_files) == 8
    assert len(result_files) == 9
    assert "01_tourism_flow_anchor_register.csv" in data_files
    assert "02_daily_driver_panel_2015_2026.csv" in data_files
    assert "04_poi_facility_inventory.csv" in data_files
    assert "Q1_daily_visitor_estimates.csv" in result_files
    assert "Q4_scenario_reoptimization.csv" in result_files
    assert all(path.stat().st_size > 0 for path in [*artifacts["data"], *artifacts["results"]])


def test_mainline_uses_runtime_directories_not_delivery_or_legacy_paths():
    root = Path(__file__).resolve().parents[1]
    mainline_scripts = [
        "build_search_theme_features.py", "build_pressure_baseline.py",
        "build_hierarchical_censored_pressure.py", "build_censored_likelihood_visitor_scale.py",
        "build_q1_timeseries_analysis.py", "build_q2_forecasts.py",
        "build_q3_absolute_resource_plan.py", "build_q4_scenario_analysis.py",
    ]
    content = "\n".join((root / "scripts" / name).read_text(encoding="utf-8") for name in mainline_scripts)
    assert "data/runtime" in content or '"runtime" / "clean"' in content
    assert "outputs/runtime" in content
    assert "data/clean" not in content
    assert "data/model_input" not in content
    assert "outputs/question1_analysis" not in content

