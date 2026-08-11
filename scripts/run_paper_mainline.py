"""Run only the four-question paper mainline; archived diagnostics stay optional."""

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.paper_mainline import PAPER_MAINLINE_SCRIPTS
from src.utils.delivery_status import assess_delivery_status, fallback_after_failed_script, freeze_verified_artifacts, write_delivery_status


def main() -> None:
    current_script = "not_started"
    try:
        for index, name in enumerate(PAPER_MAINLINE_SCRIPTS, start=1):
            current_script = name
            print(f"[{index}/{len(PAPER_MAINLINE_SCRIPTS)}] running {name}")
            subprocess.run([sys.executable, str(ROOT / "scripts" / name)], check=True)
    except subprocess.CalledProcessError:
        status = fallback_after_failed_script(ROOT, current_script)
        write_delivery_status(ROOT, status)
        print(f"[FALLBACK] {status['delivery_mode']}: {status['missing_questions']}")
        raise
    status = assess_delivery_status(ROOT)
    if status["delivery_mode"] != "full_model":
        write_delivery_status(ROOT, status)
        raise RuntimeError(f"mainline completed but delivery checkpoints failed: {status['missing_questions']}")
    freeze_verified_artifacts(ROOT)
    write_delivery_status(ROOT, status)
    print("[OK] paper mainline completed; verified Q1-Q4 artifact snapshot refreshed.")


if __name__ == "__main__":
    main()
