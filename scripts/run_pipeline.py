"""运行最小竞赛建模流水线，验证配置、输出与图表路径。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.validation import validate_dataframe
from src.utils.config import load_config
from src.viz.plots import save_example_figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the competition-modeling skeleton.")
    parser.add_argument("--csv", type=Path, help="Optional CSV file to validate.")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config.yaml")
    for output_path in config["paths"].values():
        Path(output_path).mkdir(parents=True, exist_ok=True)

    manifest = {
        "project_name": config["project_name"],
        "random_seed": config["random_seed"],
        "model_name": config["model"]["name"],
    }
    results_path = Path(config["paths"]["results"])
    (results_path / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_example_figure(Path(config["paths"]["figures"]) / "example_figure.png")

    if args.csv:
        report = validate_dataframe(pd.read_csv(args.csv), required_columns=[])
        (results_path / "data_validation.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
