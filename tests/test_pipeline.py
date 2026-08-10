import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_writes_manifest() -> None:
    subprocess.run([sys.executable, "scripts/run_pipeline.py"], check=True)

    manifest = Path("outputs/results/run_manifest.json")
    assert json.loads(manifest.read_text(encoding="utf-8"))["project_name"] == "MCM_2026"
