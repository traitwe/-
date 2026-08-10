"""Run only the four-question paper mainline; archived diagnostics stay optional."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paper_mainline import PAPER_MAINLINE_SCRIPTS


def main() -> None:
    for index, name in enumerate(PAPER_MAINLINE_SCRIPTS, start=1):
        print(f"[{index}/{len(PAPER_MAINLINE_SCRIPTS)}] running {name}")
        subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)
    print("[OK] paper mainline completed; relative-tier and Pareto scripts are archived diagnostics.")


if __name__ == "__main__":
    main()
