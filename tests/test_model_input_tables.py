import pandas as pd
import subprocess
import sys
from pathlib import Path

from src.features.model_inputs import build_model_input_tables


def test_model_input_tables_keep_region_codes_and_report_censoring():
    flow = pd.DataFrame(
        {
            "date": ["2024-07-01", "2024-07-02"],
            "attraction_name": ["景点甲", "景点乙"],
            "region_code": ["BDH", "SHG"],
            "visitor_index": [3.0, 4.0],
            "daily_rank": [1, 2],
        }
    )
    anchors = pd.DataFrame(
        {
            "period_start": ["2024-07-01", "2024-07-01"],
            "period_end": ["2024-07-01", "2024-07-01"],
            "region_name": ["Beidaihe_District", "Qinhuangdao_City"],
            "visitor_count_10k_persons": [10.0, 20.0],
            "frequency": ["daily_peak", "daily_peak"],
            "data_status": ["verified", "verified"],
            "value_qualifier": ["exact", "exact"],
            "source_dataset": ["regional", "city"],
        }
    )

    panel, prepared_anchors, quality = build_model_input_tables(
        flow,
        anchors,
        region_map={"Beidaihe_District": "BDH", "Qinhuangdao_City": "CITY"},
    )

    assert len(panel) == 4
    assert set(prepared_anchors["region_code"]) == {"BDH", "CITY"}
    assert set(prepared_anchors["source_dataset"]) == {"regional", "city"}
    assert quality["panel_rows"] == 4
    assert quality["censored_rows"] == 2
    assert quality["anchor_rows"] == 2


def test_build_script_runs_from_project_root():
    project_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "scripts/build_b_model_inputs.py"],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
