"""Build the lean runnable delivery programs and paper-facing core-code excerpts."""

from __future__ import annotations

import ast
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


PROGRAMS: dict[str, list[str]] = {
    "question1_model.py": [
        "src/models/pressure_baseline.py",
        "src/models/hierarchical_censored_pressure.py",
        "src/models/scale_calibration.py",
        "src/analysis/q1_timeseries.py",
        "scripts/build_pressure_baseline.py",
        "scripts/build_hierarchical_censored_pressure.py",
        "scripts/build_censored_likelihood_visitor_scale.py",
        "scripts/build_q1_timeseries_analysis.py",
    ],
    "question2_model.py": [
        "src/models/question2_forecasting.py",
        "src/analysis/q2_forecast_outputs.py",
        "scripts/build_q2_forecasts.py",
    ],
    "question3_model.py": [
        "src/models/question3_absolute_capacity.py",
        "src/models/question3_staff_scheduling.py",
        "src/models/question3_absolute_optimizer.py",
        "src/models/value_of_time.py",
        "src/analysis/q3_absolute_resource_outputs.py",
        "src/analysis/vot_outputs.py",
        "scripts/build_q3_absolute_resource_plan.py",
    ],
    "question4_model.py": [
        "src/models/question4_scenarios.py",
        "src/analysis/q4_scenario_outputs.py",
        "scripts/build_q4_scenario_analysis.py",
    ],
}


BOOTSTRAP = r'''# Inputs: data/submission/01--08; outputs: outputs/submission/Q1--Q4.
from pathlib import Path
import pandas as pd


def _submission_frame(root: Path, filename: str, source_table: str | None = None, drop_source: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(root / "data" / "submission" / filename, encoding="utf-8-sig")
    if source_table is not None:
        frame = frame.loc[frame["source_table"].eq(source_table)].copy()
    return frame.drop(columns=["source_table"], errors="ignore") if drop_source else frame


def _write_runtime(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def prepare_submission_runtime(root: Path) -> None:
    """Materialise the small runtime views required by the four final programs."""
    clean = root / "data" / "runtime" / "clean"
    model = root / "data" / "runtime" / "model_input"
    clean.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    drivers = _submission_frame(root, "02_daily_driver_panel_2015_2026.csv")
    _write_runtime(drivers, clean / "calendar_2015_2026.csv")
    _write_runtime(drivers, clean / "daily_weather_2015_2025.csv")
    _write_runtime(drivers, model / "daily_search_theme_features_2016_2026.csv")
    _write_runtime(_submission_frame(root, "03_attraction_observation_panel.csv", drop_source=True), model / "censored_attraction_observation_panel.csv")
    flow = _submission_frame(root, "01_tourism_flow_anchor_register.csv")
    _write_runtime(flow.loc[flow.get("source_table", pd.Series(dtype=str)).eq("annual_city_tourism_1999_2024.csv")].drop(columns=["source_table"], errors="ignore"), clean / "annual_city_tourism_1999_2024.csv")
    _write_runtime(flow.loc[flow.get("source_table", pd.Series(dtype=str)).eq("qinhuangdao_summer_regional_flow_anchors_2023_2025.csv")].drop(columns=["source_table"], errors="ignore"), clean / "raw_cleaned" / "qinhuangdao_summer_regional_flow_anchors_2023_2025.csv")
    quality = _submission_frame(root, "08_data_source_quality_register.csv")
    _write_runtime(quality.loc[quality.get("source_table", pd.Series(dtype=str)).eq("calibration_anchor_scope_ledger.csv")].drop(columns=["source_table"], errors="ignore"), model / "calibration_anchor_scope_ledger.csv")
    resources = _submission_frame(root, "05_resource_constraint_register.csv")
    _write_runtime(resources.loc[resources.get("source_table", pd.Series(dtype=str)).eq("hourly_service_time_evidence.csv")].drop(columns=["source_table"], errors="ignore"), clean / "hourly_service_time_evidence.csv")
    parameters = _submission_frame(root, "07_model_parameter_scenarios.csv")
    _write_runtime(parameters.loc[parameters.get("source_table", pd.Series(dtype=str)).eq("q3_absolute_planning_scenarios.csv")].drop(columns=["source_table"], errors="ignore"), model / "q3_absolute_planning_scenarios.csv")
    _write_runtime(parameters.loc[parameters.get("source_table", pd.Series(dtype=str)).eq("q3_vot_scenarios.csv")].drop(columns=["source_table"], errors="ignore"), model / "q3_vot_scenarios.csv")


def _export_submission(root: Path, filename: str, runtime_files: list[str]) -> None:
    frames = []
    for relative in runtime_files:
        frame = pd.read_csv(root / relative, encoding="utf-8-sig")
        frame.insert(0, "source_table", Path(relative).name)
        frames.append(frame)
    output = root / "outputs" / "submission" / filename
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True, sort=False).to_csv(output, index=False, encoding="utf-8-sig")
'''


EXPORTS: dict[str, str] = {
    "question1_model.py": r'''
if __name__ == "__main__":
    _export_submission(ROOT, "Q1_daily_visitor_estimates.csv", [
        "outputs/runtime/question1_analysis/q1_city_daily_annual_constrained.csv",
        "outputs/runtime/question1_analysis/q1_regional_pressure_decomposition.csv",
    ])
    _export_submission(ROOT, "Q1_validation_summary.csv", [
        "outputs/runtime/question1_analysis/q1_factor_strength.csv",
        "outputs/runtime/question1_analysis/q1_regional_season_classification.csv",
    ])
''',
    "question2_model.py": r'''
if __name__ == "__main__":
    _export_submission(ROOT, "Q2_2026_forecast.csv", [
        "outputs/runtime/question2_analysis/q2_city_monthly_forecast_2026.csv",
        "outputs/runtime/question2_analysis/q2_region_daily_forecast_2026.csv",
    ])
    _export_submission(ROOT, "Q2_validation_summary.csv", [
        "outputs/runtime/question2_analysis/q2_annual_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_daily_model_comparison.csv",
        "outputs/runtime/question2_analysis/q2_daily_rolling_origin_validation.csv",
        "outputs/runtime/question2_analysis/q2_daily_conformal_calibration_2026.csv",
        "outputs/runtime/question2_analysis/q2_external_anchor_validation.csv",
    ])
''',
    "question3_model.py": r'''
if __name__ == "__main__":
    _export_submission(ROOT, "Q3_daily_resource_plan.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_daily_optimized_plan_2026.csv",
    ])
    _export_submission(ROOT, "Q3_seasonal_recommendations.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_seasonal_resource_plan_2026.csv",
        "outputs/runtime/question3_analysis/q3_beidaihe_absolute_summer_temporary_plan_2026.csv",
    ])
    _export_submission(ROOT, "Q3_sensitivity_summary.csv", [
        "outputs/runtime/question3_analysis/q3_absolute_resource_sensitivity_2026.csv",
        "outputs/runtime/question3_analysis/q3_absolute_cost_experience_pareto_2026.csv",
        "outputs/runtime/question3_analysis/q3_bdh_temporary_capacity_relaxation.csv",
    ])
''',
    "question4_model.py": r'''
if __name__ == "__main__":
    _export_submission(ROOT, "Q4_scenario_reoptimization.csv", [
        "outputs/runtime/question4_analysis/q4_event_counterfactual_2026.csv",
        "outputs/runtime/question4_analysis/q4_rain_counterfactual_2026.csv",
    ])
    _export_submission(ROOT, "Q4_robustness_summary.csv", [
        "outputs/runtime/question4_analysis/q4_baseline_reoptimization_summary_2026.csv",
        "outputs/runtime/question4_analysis/q4_robustness_sensitivity_ranking.csv",
        "outputs/runtime/question4_analysis/q4_scenario_window_adjustment_summary.csv",
        "outputs/runtime/question4_analysis/q4_lambda_tradeoff_2026.csv",
    ])
''',
}


APPENDIX_CORE: dict[str, str] = {
    "question1_core.py": '''# 问题一：状态划分、左删失似然与游客规模标定
import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

def censored_negative_log_posterior(beta, x, observed, threshold, censored):
    """排名上榜值按正态密度计，未上榜值按左删失概率计。"""
    mean = x @ beta[:-1]
    sigma = max(beta[-1], 1e-6)
    observed_part = -np.log(np.maximum(
        np.exp(-0.5 * ((observed - mean) / sigma) ** 2) / sigma, 1e-12)).sum()
    censored_part = -log_ndtr((threshold - mean[censored]) / sigma).sum()
    return observed_part + censored_part + 0.1 * np.square(beta[:-1]).sum()

def estimate_pressure(x, observed, threshold, censored):
    initial = np.r_[np.zeros(x.shape[1]), 1.0]
    result = minimize(censored_negative_log_posterior, initial,
                      args=(x, observed, threshold, censored), method="L-BFGS-B")
    return x @ result.x[:-1]

def calibrate_visitors(pressure, anchor_pressure, anchor_visitors):
    scale = np.median(anchor_visitors / np.maximum(anchor_pressure, 1e-8))
    return np.maximum(pressure * scale, 0.0)
''',
    "question2_core.py": '''# 问题二：年总量—月度份额—日尺度协变量预测
import numpy as np

def fit_ridge(x, y, penalty=1.0):
    x1 = np.c_[np.ones(len(x)), x]
    return np.linalg.solve(x1.T @ x1 + penalty * np.eye(x1.shape[1]), x1.T @ y)

def forecast_daily(train_x, train_y, target_x, annual_total, month_share):
    beta = fit_ridge(train_x, np.log1p(train_y))
    raw = np.expm1(np.c_[np.ones(len(target_x)), target_x] @ beta).clip(0)
    raw = raw * np.asarray(month_share)
    # 在每月内保持协变量造成的相对起伏，并与年度总量一致。
    return annual_total * raw / max(raw.sum(), 1e-8)

def split_conformal_interval(prediction, residuals, level=0.90):
    radius = np.quantile(np.abs(residuals), level)
    return prediction - radius, prediction + radius
''',
    "question3_core.py": '''# 问题三：停车—接驳—排班联合资源配置
import math
from itertools import product

def resource_plan(visitors, car_share, dwell_hours, turnover, vehicle_capacity,
                  trips_per_vehicle, staff_rate, permanent_spaces, temporary_spaces):
    parking_need = math.ceil(visitors * car_share * dwell_hours / max(turnover, 1e-8))
    shuttle_need = math.ceil(visitors / max(vehicle_capacity * trips_per_vehicle, 1))
    staff_need = math.ceil(visitors / max(staff_rate, 1))
    candidates = []
    for temporary, shuttles, staff in product(range(temporary_spaces + 1),
                                              range(shuttle_need + 4),
                                              range(staff_need + 4)):
        shortage = max(parking_need - permanent_spaces - temporary, 0) + \\
                   max(shuttle_need - shuttles, 0) + max(staff_need - staff, 0)
        cost = temporary + 2 * shuttles + 0.6 * staff
        candidates.append((cost + 4 * shortage, temporary, shuttles, staff, shortage))
    _, temporary, shuttles, staff, shortage = min(candidates)
    return {"temporary_spaces": temporary, "shuttle_vehicles": shuttles,
            "staff_shifts": staff, "service_shortfall": shortage}
''',
    "question4_core.py": '''# 问题四：客流冲击、资源重优化与稳健性比较
import pandas as pd

def apply_event_pulse(frame, visitor_column, event_date, intensity):
    result = frame.copy()
    offset = (pd.to_datetime(result["date"]) - pd.Timestamp(event_date)).dt.days
    pulse = offset.map({-2: .25, -1: .50, 0: 1., 1: .50, 2: .25}).fillna(0.)
    result["scenario_visitors"] = result[visitor_column] * (1 + intensity * pulse)
    return result

def apply_continuous_rain(frame, visitor_column, start_date, days, attenuation):
    result = frame.copy()
    offset = (pd.to_datetime(result["date"]) - pd.Timestamp(start_date)).dt.days
    factor = (1 - attenuation * (offset + 1)).clip(lower=0)
    result["scenario_visitors"] = result[visitor_column] * factor.where(offset.between(0, days - 1), 1.)
    return result

def compare_plan(baseline, shocked, scenario, resource_plan):
    plans = [resource_plan(v, **scenario) for v in shocked["scenario_visitors"]]
    return baseline.assign(reoptimized_plan=plans)
''',
}


# Appendix code must be taken verbatim from the real model implementation.
# Never use the earlier pedagogical examples as paper-facing source code.
APPENDIX_FUNCTIONS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "question1_core.py": (
        ("src/models/hierarchical_censored_pressure.py", ("negative_log_posterior",)),
        ("src/models/scale_calibration.py", ("calibrate_daily_visitor_scale",)),
    ),
    "question2_core.py": (
        ("src/models/question2_forecasting.py", ("fit_daily_ridge_forecaster",)),
        ("src/analysis/q2_forecast_outputs.py", ("apply_regional_relative_split_conformal_interval",)),
    ),
    "question3_core.py": (
        ("src/models/question3_absolute_optimizer.py", ("optimize_absolute_resources",)),
    ),
    "question4_core.py": (
        ("src/models/question4_scenarios.py", ("apply_event_pulse", "apply_continuous_rain")),
        ("src/analysis/q4_scenario_outputs.py", ("compare_baseline_with_reoptimisation",)),
    ),
}


def _local_import_line_ranges(source: str) -> set[int]:
    tree = ast.parse(source)
    remove: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module == "__future__" or (node.module and node.module.startswith("src."))):
            remove.update(range(node.lineno, node.end_lineno + 1))
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            func = node.value.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "insert"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "path"
                and isinstance(func.value.value, ast.Name)
                and func.value.value.id == "sys"
            ):
                remove.update(range(node.lineno, node.end_lineno + 1))
    return remove


def _inline_source(root: Path, relative_path: str) -> str:
    source = (root / relative_path).read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    remove = _local_import_line_ranges(source)
    retained = ["" if index in remove else line for index, line in enumerate(lines, start=1)]
    return "\n\n" + "\n".join(retained) + "\n"


def _extract_real_functions(root: Path, relative_path: str, function_names: tuple[str, ...]) -> str:
    """Return verbatim top-level functions from the actual model implementation."""
    source = (root / relative_path).read_text(encoding="utf-8-sig")
    lines = source.splitlines()
    functions = {
        node.name: node for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    missing = set(function_names).difference(functions)
    if missing:
        raise ValueError(f"{relative_path} is missing appendix functions: {sorted(missing)}")
    return "\n\n".join(
        "\n".join(lines[functions[name].lineno - 1:functions[name].end_lineno])
        for name in function_names
    )


def _build_appendix_excerpt(root: Path, filename: str) -> str:
    """Build a paper excerpt only from functions executed by the full model."""
    parts = [
        f"# {filename}: 以下函数从完整运行程序的真实实现原样摘取。",
        "# 完整输入、输出与辅助函数见支撑材料中的 question*_model.py。",
    ]
    for relative_path, function_names in APPENDIX_FUNCTIONS[filename]:
        parts.append(_extract_real_functions(root, relative_path, function_names))
    return "\n\n".join(parts) + "\n"


def build_final_programs(root: Path = PROJECT_ROOT, output_directory: Path | None = None) -> list[Path]:
    """Write four runnable entry points, a shared runtime, and four short appendix excerpts."""
    root = root.resolve()
    output = output_directory or root / "final_programs"
    output.mkdir(parents=True, exist_ok=True)
    common_path = output / "common_runtime.py"
    common_path.write_text('"""Shared data materialisation and delivery-table export utilities."""\n' + BOOTSTRAP, encoding="utf-8")
    paths: list[Path] = [common_path]
    for filename, sources in PROGRAMS.items():
        chunks = [
            '"""Runnable final program for one modelling question."""\n',
            'from pathlib import Path\n',
            'from common_runtime import _export_submission, prepare_submission_runtime\n',
        ]
        if filename == "question4_model.py":
            chunks.append(
                'from question3_model import (attach_vot_costs, build_daily_absolute_plan, '\
                'optimize_absolute_resources)\n'
            )
        module_sources = [source for source in sources if source.startswith("src/")]
        script_sources = [source for source in sources if source.startswith("scripts/")]
        chunks.extend(_inline_source(root, source) for source in module_sources)
        chunks.append("\nif __name__ == \"__main__\":\n    prepare_submission_runtime(Path(__file__).resolve().parents[1])\n")
        chunks.extend(_inline_source(root, source) for source in script_sources)
        chunks.append(EXPORTS[filename])
        path = output / filename
        path.write_text("".join(chunks), encoding="utf-8")
        paths.append(path)
    excerpt_directory = output / "appendix_core"
    excerpt_directory.mkdir(parents=True, exist_ok=True)
    for filename in APPENDIX_FUNCTIONS:
        (excerpt_directory / filename).write_text(_build_appendix_excerpt(root, filename), encoding="utf-8")
    return paths


if __name__ == "__main__":
    for program in build_final_programs():
        print(program)
