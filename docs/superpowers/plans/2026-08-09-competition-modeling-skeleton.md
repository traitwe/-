# Competition Modeling Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a clean, runnable and reusable Python skeleton for a mathematical-modeling competition project.

**Architecture:** Raw data remains immutable under `data/raw`; data validation and configuration are reusable Python modules under `src`. A minimal command-line pipeline validates an optional input dataset, writes a run manifest, and produces an example figure so the end-to-end output path is verifiable before a contest problem is known.

**Tech Stack:** Python 3.12, pandas, NumPy, Matplotlib, PyYAML, pytest.

---

## File structure

- `README.md`: setup and contest workflow.
- `.gitignore`: excludes local environments, datasets, and generated outputs while retaining folder placeholders.
- `requirements.txt`: minimal reproducible dependency list.
- `config.yaml`: central paths, seed, and placeholder modeling parameters.
- `src/utils/config.py`: typed configuration loader and output directory setup.
- `src/data/validation.py`: CSV validation report for missing columns, empty data, duplicates, and null rates.
- `src/viz/plots.py`: single reusable line-chart helper for competition figures.
- `scripts/run_pipeline.py`: command-line entry point that creates a manifest and optional validation report/example figure.
- `tests/`: tests for configuration, validation, plotting, and pipeline execution.
- `data/`, `notebooks/`, `outputs/`, `docs/`: empty but intentionally versioned workflow locations.

### Task 1: Establish project metadata and folders

**Files:**
- Create: `README.md`, `.gitignore`, `requirements.txt`, `config.yaml`
- Create: `data/raw/.gitkeep`, `data/processed/.gitkeep`, `notebooks/.gitkeep`, `outputs/figures/.gitkeep`, `outputs/tables/.gitkeep`, `outputs/results/.gitkeep`, `docs/.gitkeep`

- [ ] **Step 1: Add the dependency list**

```text
numpy>=2.0,<3.0
pandas>=2.2,<3.0
matplotlib>=3.9,<4.0
PyYAML>=6.0,<7.0
pytest>=8.0,<9.0
```

- [ ] **Step 2: Add the central configuration**

```yaml
project_name: MCM_2026
random_seed: 2026
paths:
  raw_data: data/raw
  processed_data: data/processed
  figures: outputs/figures
  tables: outputs/tables
  results: outputs/results
model:
  name: placeholder
  parameters: {}
```

- [ ] **Step 3: Add ignore rules and workflow documentation**

The ignore file keeps `.venv/`, `__pycache__/`, `*.pyc`, `data/raw/*`, `data/processed/*`, and `outputs/**/*` out of Git while explicitly retaining each `.gitkeep`. The README documents environment creation, dependency installation, `python scripts/run_pipeline.py`, and the raw → processed → model → analysis → figure/table workflow.

- [ ] **Step 4: Verify expected paths exist**

Run: `Get-ChildItem -Recurse -Force`

Expected: all paths listed above are present and no dataset or generated result is created.

### Task 2: Add configuration support using test-driven development

**Files:**
- Create: `src/__init__.py`, `src/utils/__init__.py`, `src/utils/config.py`
- Create: `tests/__init__.py`, `tests/test_config.py`

- [ ] **Step 1: Write the failing configuration test**

```python
from pathlib import Path
from src.utils.config import load_config


def test_load_config_resolves_project_relative_paths() -> None:
    config = load_config(Path("config.yaml"))
    assert config["random_seed"] == 2026
    assert config["paths"]["figures"].is_absolute()
    assert config["paths"]["figures"].name == "figures"
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `python -m pytest tests/test_config.py -v`

Expected: FAIL because `src.utils.config` does not yet exist.

- [ ] **Step 3: Implement `load_config`**

```python
from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path) -> dict[str, Any]:
    config_path = path.resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    for key, value in config["paths"].items():
        config["paths"][key] = (config_path.parent / value).resolve()
    return config
```

- [ ] **Step 4: Re-run the test**

Run: `python -m pytest tests/test_config.py -v`

Expected: PASS.

### Task 3: Add reusable CSV validation

**Files:**
- Create: `src/data/__init__.py`, `src/data/validation.py`, `tests/test_validation.py`

- [ ] **Step 1: Write the failing validation test**

```python
import pandas as pd
from src.data.validation import validate_dataframe


def test_validation_counts_rows_duplicates_and_missing_columns() -> None:
    frame = pd.DataFrame({"x": [1, 1], "y": [None, 2]})
    report = validate_dataframe(frame, required_columns=["x", "z"])
    assert report["row_count"] == 2
    assert report["duplicate_row_count"] == 0
    assert report["missing_required_columns"] == ["z"]
    assert report["null_rates"]["y"] == 0.5
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `python -m pytest tests/test_validation.py -v`

Expected: FAIL because `validate_dataframe` does not yet exist.

- [ ] **Step 3: Implement the validation report**

```python
from collections.abc import Sequence
from typing import Any

import pandas as pd


def validate_dataframe(frame: pd.DataFrame, required_columns: Sequence[str]) -> dict[str, Any]:
    missing = [column for column in required_columns if column not in frame.columns]
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "missing_required_columns": missing,
        "null_rates": {column: float(rate) for column, rate in frame.isna().mean().items()},
    }
```

- [ ] **Step 4: Re-run the test**

Run: `python -m pytest tests/test_validation.py -v`

Expected: PASS.

### Task 4: Add figure output and a minimal pipeline

**Files:**
- Create: `src/viz/__init__.py`, `src/viz/plots.py`, `scripts/run_pipeline.py`
- Create: `tests/test_plots.py`, `tests/test_pipeline.py`

- [ ] **Step 1: Write the failing plotting test**

```python
from pathlib import Path
from src.viz.plots import save_example_figure


def test_save_example_figure_creates_png(tmp_path: Path) -> None:
    output = tmp_path / "example.png"
    save_example_figure(output)
    assert output.is_file()
    assert output.stat().st_size > 0
```

- [ ] **Step 2: Implement a deterministic figure helper**

```python
from pathlib import Path

import matplotlib.pyplot as plt


def save_example_figure(output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([1, 2, 3], [1, 4, 9], marker="o")
    axis.set(xlabel="x", ylabel="y", title="Example competition figure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
```

- [ ] **Step 3: Write the failing pipeline test**

```python
import json
import subprocess
import sys
from pathlib import Path


def test_pipeline_writes_manifest() -> None:
    subprocess.run([sys.executable, "scripts/run_pipeline.py"], check=True)
    manifest = Path("outputs/results/run_manifest.json")
    assert json.loads(manifest.read_text(encoding="utf-8"))["project_name"] == "MCM_2026"
```

- [ ] **Step 4: Implement the command-line pipeline**

The script loads `config.yaml`, creates every configured output directory, writes `outputs/results/run_manifest.json` containing the project name and random seed, and calls `save_example_figure` for `outputs/figures/example_figure.png`. It accepts an optional `--csv` path; if supplied, it writes `outputs/results/data_validation.json` from `validate_dataframe`.

- [ ] **Step 5: Run the complete test suite**

Run: `python -m pytest -v`

Expected: PASS for configuration, validation, plotting, and pipeline tests.

### Task 5: Verify the contest-ready workflow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the pipeline manually**

Run: `python scripts/run_pipeline.py`

Expected: `outputs/results/run_manifest.json` and `outputs/figures/example_figure.png` are created.

- [ ] **Step 2: Confirm generated output remains ignored by Git**

Run: `git status --short`

Expected: If the user initializes Git later, generated output files are ignored while `.gitkeep`, source, tests, documentation, configuration, and requirements remain trackable.

- [ ] **Step 3: Document the verification command in the README**

Add `python -m pytest -v` under a “验证” heading and state that contest-specific models should be added in `src/models/` rather than scripts or notebooks.
