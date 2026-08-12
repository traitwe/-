"""Fit and export the first transparent relative-pressure baseline."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.pressure_baseline import fit_pressure_baseline, rolling_time_validation


def main() -> None:
    model_input_dir = PROJECT_ROOT / "data" / "runtime" / "model_input"
    clean_dir = PROJECT_ROOT / "data" / "runtime" / "clean"
    panel = pd.read_csv(model_input_dir / "censored_attraction_observation_panel.csv")
    calendar = pd.read_csv(clean_dir / "calendar_2015_2026.csv")
    weather = pd.read_csv(clean_dir / "daily_weather_2015_2025.csv")
    search = pd.read_csv(model_input_dir / "daily_search_theme_features_2016_2026.csv")
    calendar_dates = pd.to_datetime(calendar["date"], errors="coerce")
    weather_dates = pd.to_datetime(weather["date"], errors="coerce")
    start_date = max(calendar_dates.min(), pd.Timestamp("2023-01-01"))
    end_date = min(calendar_dates.max(), weather_dates.max(), pd.Timestamp("2025-12-31"))
    region_codes = ["BDH", "SHG", "HGA"]
    prediction_frame = pd.MultiIndex.from_product(
        [pd.date_range(start_date, end_date, freq="D"), region_codes],
        names=["date", "region_code"],
    ).to_frame(index=False)
    pressure = fit_pressure_baseline(
        panel,
        calendar,
        weather,
        search,
        prediction_frame=prediction_frame,
    )
    pressure.to_csv(
        model_input_dir / "daily_region_pressure_baseline_2023_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rolling_scores = rolling_time_validation(
        panel,
        calendar,
        weather,
        search,
        cutoffs=["2023-06-30", "2024-06-30"],
    )
    rolling_scores.to_csv(
        model_input_dir / "pressure_baseline_rolling_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality = {
        "rows": int(len(pressure)),
        "date_start": str(pressure["date"].min().date()),
        "date_end": str(pressure["date"].max().date()),
        "region_codes": sorted(pressure["region_code"].unique().tolist()),
        "estimate_label": "relative_pressure_index",
        "rolling_validation_rows": int(len(rolling_scores)),
        "rolling_mae": rolling_scores["mae"].round(6).tolist(),
        "note": "Relative pressure only; no row is an observed or estimated absolute visitor count.",
    }
    (model_input_dir / "pressure_baseline_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
