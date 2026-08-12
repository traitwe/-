"""Checkpointed paper-delivery state with a recoverable verified-artifact path."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


REQUIRED_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "Q1": (
        "outputs/runtime/question1_analysis/q1_city_daily_annual_constrained.csv",
        "outputs/runtime/question1_analysis/q1_analysis_quality_report.json",
    ),
    "Q2": (
        "outputs/runtime/question2_analysis/q2_annual_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_region_daily_forecast_2026.csv",
        "outputs/runtime/question2_analysis/q2_daily_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_daily_conformal_calibration_2026.csv",
    ),
    "Q3": (
        "outputs/runtime/question3_analysis/q3_absolute_daily_resource_plan_2026.csv",
        "outputs/runtime/question3_analysis/q3_absolute_daily_optimized_plan_2026.csv",
        "outputs/runtime/question3_analysis/q3_absolute_resource_sensitivity_2026.csv",
    ),
    "Q4": (
        "outputs/runtime/question4_analysis/q4_event_counterfactual_2026.csv",
        "outputs/runtime/question4_analysis/q4_rain_counterfactual_2026.csv",
        "outputs/runtime/question4_analysis/q4_baseline_reoptimization_summary_2026.csv",
        "outputs/runtime/question4_analysis/q4_robustness_sensitivity_ranking.csv",
    ),
}


def _missing_by_question(root: Path) -> dict[str, list[str]]:
    return {
        question: [relative for relative in files if not (root / relative).is_file()]
        for question, files in REQUIRED_ARTIFACTS.items()
        if any(not (root / relative).is_file() for relative in files)
    }


def assess_delivery_status(root: Path) -> dict[str, object]:
    """Assess current outputs without changing them."""
    root = Path(root)
    missing = _missing_by_question(root)
    snapshot = root / "outputs" / "verified_artifacts"
    if not missing:
        mode = "full_model"
        delivery_root = "outputs"
        scope = "Q1-Q4 full model outputs are present."
    elif snapshot.is_dir():
        mode = "verified_artifact_fallback"
        delivery_root = "outputs/verified_artifacts"
        scope = "Use only the last verified snapshot; do not combine it with partially regenerated outputs."
    else:
        mode = "no_deliverable_snapshot"
        delivery_root = None
        scope = "No complete current output and no verified snapshot are available."
    return {
        "delivery_mode": mode,
        "delivery_root": delivery_root,
        "missing_questions": sorted(missing),
        "missing_artifacts": missing,
        "scope_note": scope,
    }


def freeze_verified_artifacts(root: Path) -> Path:
    """Replace the verified snapshot only after all four checkpoints pass."""
    root = Path(root)
    status = assess_delivery_status(root)
    if status["delivery_mode"] != "full_model":
        raise ValueError("cannot freeze artifacts before all Q1-Q4 checkpoints pass")
    snapshot = root / "outputs" / "verified_artifacts"
    if snapshot.exists():
        shutil.rmtree(snapshot)
    for directory in ("question1_analysis", "question2_analysis", "question3_analysis", "question4_analysis"):
        shutil.copytree(root / "outputs" / "runtime" / directory, snapshot / directory)
    return snapshot


def fallback_after_failed_script(root: Path, failed_script: str) -> dict[str, object]:
    """Force the verified-snapshot mode after any mainline script failure."""
    root = Path(root)
    snapshot = root / "outputs" / "verified_artifacts"
    if snapshot.is_dir():
        mode = "verified_artifact_fallback"
        delivery_root: str | None = "outputs/verified_artifacts"
        scope = "A mainline script failed. Use only the prior verified snapshot; current outputs may be partial."
    else:
        mode = "no_deliverable_snapshot"
        delivery_root = None
        scope = "A mainline script failed and no prior verified snapshot is available."
    return {
        "delivery_mode": mode,
        "delivery_root": delivery_root,
        "missing_questions": [],
        "missing_artifacts": {},
        "failed_script": failed_script,
        "scope_note": scope,
    }


def write_delivery_status(root: Path, status: dict[str, object] | None = None) -> Path:
    """Persist an auditable delivery decision for the paper handoff."""
    root = Path(root)
    payload = dict(status or assess_delivery_status(root))
    payload["generated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    destination = root / "outputs" / "delivery_status.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination

