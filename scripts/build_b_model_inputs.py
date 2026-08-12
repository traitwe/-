"""Build traceable inputs for the Qinhuangdao B-problem models."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.model_inputs import build_model_input_tables
from src.features.anchor_selection import classify_direct_region_anchors


CLEAN_DIR = PROJECT_ROOT / "data" / "runtime" / "clean"
ANCHOR_DIR = CLEAN_DIR / "raw_cleaned"
OUTPUT_DIR = PROJECT_ROOT / "data" / "runtime" / "model_input"

REGION_MAP = {
    "Qinhuangdao_City": "CITY",
    "Beidaihe_District": "BDH",
    "Beidaihe_District_Core": "BDH_CORE",
    "Beidaihe_New_District": "BND",
    "Shanhaiguan_District": "SHG",
    "Shanhaiguan_Scenic_Area": "SHG_SCENIC",
    "Shanhaiguan_Tourism_Area": "SHG_TOURISM",
    "Haigang_District": "HGA",
    "Aranya_Leisure_Area": "ARA",
}


def load_visitor_anchor_sources(anchor_dir: Path) -> pd.DataFrame:
    """Load only source files that actually contain visitor-count anchor fields."""
    frames: list[pd.DataFrame] = []
    for path in sorted(anchor_dir.glob("*.csv")):
        frame = pd.read_csv(path, dtype=str, on_bad_lines="skip")
        if {"period_start", "period_end", "region_name", "visitor_count_10k_persons"}.issubset(frame.columns):
            frame["source_dataset"] = path.name
            frames.append(frame)
    if not frames:
        raise ValueError("no visitor-count anchor source files found")
    return pd.concat(frames, ignore_index=True, sort=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    attraction_flow = pd.read_csv(CLEAN_DIR / "daily_attraction_flow_index_2021_2025.csv")
    visitor_anchors = load_visitor_anchor_sources(ANCHOR_DIR)
    panel, anchors, quality = build_model_input_tables(attraction_flow, visitor_anchors, REGION_MAP)
    anchor_ledger = classify_direct_region_anchors(
        anchors,
        target_regions={"BDH", "HGA", "SHG"},
        model_start="2023-01-01",
        model_end="2025-12-31",
    )

    panel.to_csv(OUTPUT_DIR / "censored_attraction_observation_panel.csv", index=False, encoding="utf-8-sig")
    anchors.to_csv(OUTPUT_DIR / "calibration_anchors_prepared.csv", index=False, encoding="utf-8-sig")
    anchor_ledger.to_csv(OUTPUT_DIR / "calibration_anchor_scope_ledger.csv", index=False, encoding="utf-8-sig")
    quality.update(
        {
            "panel_date_start": str(panel["date"].min().date()),
            "panel_date_end": str(panel["date"].max().date()),
            "panel_region_codes": sorted(panel["region_code"].unique().tolist()),
            "anchor_region_codes": sorted(anchors["region_code"].unique().tolist()),
            "anchor_source_datasets": sorted(anchors["source_dataset"].dropna().unique().tolist()),
            "anchor_scope_decisions": {
                key: int(value)
                for key, value in anchor_ledger["scope_decision"].value_counts().sort_index().items()
            },
        }
    )
    (OUTPUT_DIR / "b_model_input_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
