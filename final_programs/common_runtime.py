"""Shared data materialisation and delivery-table export utilities."""
# Inputs: data/submission/01--08; outputs: outputs/submission/Q1--Q4.
from pathlib import Path
import pandas as pd


def _submission_frame(root: Path, filename: str, source_table: str | None = None, drop_source: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(root / "data" / "submission" / filename, encoding="utf-8-sig")
    if source_table is not None:
        frame = frame.loc[frame["source_table"].eq(source_table)].copy()
    return frame.drop(columns=["source_table"], errors="ignore") if drop_source else frame


def _write_runtime(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def prepare_submission_runtime(root: Path) -> None:
    """Materialise the small runtime views required by the four final programs."""
    clean = root / "data" / "runtime" / "clean"
    model = root / "data" / "runtime" / "model_input"
    clean.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    drivers = _submission_frame(root, "02_daily_driver_panel_2015_2026.csv")
    _write_runtime(drivers, clean / "calendar_2015_2026.csv")
    _write_runtime(drivers, clean / "daily_weather_2015_2025.csv")
    _write_runtime(drivers, model / "daily_search_theme_features_2016_2026.csv")
    _write_runtime(_submission_frame(root, "03_attraction_observation_panel.csv", drop_source=True), model / "censored_attraction_observation_panel.csv")
    flow = _submission_frame(root, "01_tourism_flow_anchor_register.csv")
    _write_runtime(flow.loc[flow.get("source_table", pd.Series(dtype=str)).eq("annual_city_tourism_1999_2024.csv")].drop(columns=["source_table"], errors="ignore"), clean / "annual_city_tourism_1999_2024.csv")
    _write_runtime(flow.loc[flow.get("source_table", pd.Series(dtype=str)).eq("qinhuangdao_summer_regional_flow_anchors_2023_2025.csv")].drop(columns=["source_table"], errors="ignore"), clean / "raw_cleaned" / "qinhuangdao_summer_regional_flow_anchors_2023_2025.csv")
    quality = _submission_frame(root, "08_data_source_quality_register.csv")
    _write_runtime(quality.loc[quality.get("source_table", pd.Series(dtype=str)).eq("calibration_anchor_scope_ledger.csv")].drop(columns=["source_table"], errors="ignore"), model / "calibration_anchor_scope_ledger.csv")
    resources = _submission_frame(root, "05_resource_constraint_register.csv")
    _write_runtime(resources.loc[resources.get("source_table", pd.Series(dtype=str)).eq("hourly_service_time_evidence.csv")].drop(columns=["source_table"], errors="ignore"), clean / "hourly_service_time_evidence.csv")
    parameters = _submission_frame(root, "07_model_parameter_scenarios.csv")
    _write_runtime(parameters.loc[parameters.get("source_table", pd.Series(dtype=str)).eq("q3_absolute_planning_scenarios.csv")].drop(columns=["source_table"], errors="ignore"), model / "q3_absolute_planning_scenarios.csv")
    _write_runtime(parameters.loc[parameters.get("source_table", pd.Series(dtype=str)).eq("q3_vot_scenarios.csv")].drop(columns=["source_table"], errors="ignore"), model / "q3_vot_scenarios.csv")


def _export_submission(root: Path, filename: str, runtime_files: list[str]) -> None:
    frames = []
    for relative in runtime_files:
        frame = pd.read_csv(root / relative, encoding="utf-8-sig")
        frame.insert(0, "source_table", Path(relative).name)
        frames.append(frame)
    output = root / "outputs" / "submission" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output, index=False, encoding="utf-8-sig")
