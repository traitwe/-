from pathlib import Path

from src.utils.delivery_status import REQUIRED_ARTIFACTS, assess_delivery_status, fallback_after_failed_script, freeze_verified_artifacts


def _create_required_outputs(root: Path) -> None:
    for question, paths in REQUIRED_ARTIFACTS.items():
        for relative in paths:
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(question, encoding="utf-8")


def test_assess_delivery_status_reports_full_model_when_all_checkpoints_exist(tmp_path):
    _create_required_outputs(tmp_path)

    status = assess_delivery_status(tmp_path)

    assert status["delivery_mode"] == "full_model"
    assert status["missing_questions"] == []


def test_assess_delivery_status_uses_verified_artifact_fallback_when_q4_missing(tmp_path):
    _create_required_outputs(tmp_path)
    freeze_verified_artifacts(tmp_path)
    for relative in REQUIRED_ARTIFACTS["Q4"]:
        (tmp_path / relative).unlink()

    status = assess_delivery_status(tmp_path)

    assert status["delivery_mode"] == "verified_artifact_fallback"
    assert status["missing_questions"] == ["Q4"]
    assert status["delivery_root"].endswith("outputs/verified_artifacts")


def test_failed_script_forces_fallback_even_when_old_outputs_still_exist(tmp_path):
    _create_required_outputs(tmp_path)
    freeze_verified_artifacts(tmp_path)

    status = fallback_after_failed_script(tmp_path, "build_q4_scenario_analysis.py")

    assert status["delivery_mode"] == "verified_artifact_fallback"
    assert status["failed_script"] == "build_q4_scenario_analysis.py"
