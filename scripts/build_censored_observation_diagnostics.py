"""Build Q1 left-censored ranking-observation diagnostics from actual sampled dates."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.censored_diagnostics import build_censored_observation_diagnostics


def main() -> None:
    model_input_dir = PROJECT_ROOT / "data" / "model_input"
    panel = pd.read_csv(model_input_dir / "censored_attraction_observation_panel.csv", encoding="utf-8-sig")
    diagnostics = build_censored_observation_diagnostics(panel)
    diagnostics.to_csv(
        model_input_dir / "censored_observation_diagnostics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    censor_rate = diagnostics["censor_rate"]
    quality = {
        "groups": int(len(diagnostics)),
        "valid_likelihood_groups": int(diagnostics["total_log_likelihood"].notna().sum()),
        "no_density_observation_groups": int(diagnostics["uncertainty_flag"].eq("no_density_observation").sum()),
        "no_positive_density_index_groups": int(diagnostics["uncertainty_flag"].eq("no_positive_density_index").sum()),
        "small_observed_sample_groups": int(diagnostics["uncertainty_flag"].eq("small_observed_sample").sum()),
        "censor_rate_summary": {
            "min": float(censor_rate.min()),
            "median": float(censor_rate.median()),
            "max": float(censor_rate.max()),
        },
        "note": "Diagnostics apply only to actual ranking sample dates; left-censored rows are not imputed visitor values.",
    }
    (model_input_dir / "censored_observation_quality_report.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(quality, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
