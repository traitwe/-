"""Export anchor-constrained daily visitor-scale scenarios for the three target regions."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.scale_calibration import calibrate_daily_visitor_scale


def main() -> None:
    model_input_dir = PROJECT_ROOT / "data" / "model_input"
    pressure = pd.read_csv(model_input_dir / "daily_region_pressure_baseline_2023_2025.csv")
    anchor_ledger = pd.read_csv(model_input_dir / "calibration_anchor_scope_ledger.csv")
    estimates, anchor_fit = calibrate_daily_visitor_scale(pressure, anchor_ledger)
    estimates.to_csv(
        model_input_dir / "daily_region_visitor_scale_estimates_2023_2025.csv",
        index=False,
        encoding="utf-8-sig",
    )
    anchor_fit.to_csv(
        model_input_dir / "visitor_scale_anchor_fit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    quality = {
        "rows": int(len(estimates)),
        "direct_anchor_fits": int(len(anchor_fit)),
        "primary_scale_anchor_fits": int(anchor_fit["calibration_role"].eq("primary_scale").sum()),
        "diagnostic_only_anchor_fits": int(anchor_fit["calibration_role"].eq("diagnostic_only").sum()),
        "calibration_regions": sorted(anchor_fit["region_code"].unique().tolist()),
        "carry_forward_rows": int(estimates["scale_interval_method"].eq("carry_forward_scenario").sum()),
        "max_abs_primary_anchor_relative_fit_error": float(
            anchor_fit.loc[anchor_fit["calibration_role"].eq("primary_scale"), "relative_fit_error"].abs().max()
        ),
        "max_abs_diagnostic_anchor_relative_fit_error": float(
            anchor_fit.loc[anchor_fit["calibration_role"].eq("diagnostic_only"), "relative_fit_error"].abs().max()
        ),
        "estimate_label": "anchor_constrained_estimate",
        "note": "Scenario estimates use annual anchors when available; conflicting short-period anchors are diagnostics, not fitting targets.",
    }
    (model_input_dir / "visitor_scale_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
