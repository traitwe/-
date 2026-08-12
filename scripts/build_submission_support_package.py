"""Build the paper appendix manifest and the matching contest support ZIP."""

from __future__ import annotations

from pathlib import Path
import csv
import zipfile

import pandas as pd


MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 仅保留直接参与 Q1--Q4 模型计算的可运行程序及其导入模块。
# 调度、交付校验、绘图排版、质量诊断、清洗和备选模型不进入竞赛支撑包。
MAINLINE_SOURCE_FILES = (
    "final_programs/common_runtime.py",
    "final_programs/question1_model.py",
    "final_programs/question2_model.py",
    "final_programs/question3_model.py",
    "final_programs/question4_model.py",
)

# 仅打包和列出当前论文正文实际引用的图件；示例图和空目录标记不属于交付物。
PAPER_FIGURE_FILES = (
    "q1_city_annual_trend.png",
    "q1_regional_peak_pressure_2025.png",
    "q2_city_monthly_forecast.png",
    "q2_summer_regional_daily_forecast.png",
    "q4_sensitivity_ranking.png",
)

SOURCE_ROLE_DESCRIPTIONS = {
    "src/__init__.py": "声明项目根包，使解压后的主线模块可被 Python 正确导入。",
    "src/analysis/__init__.py": "声明分析输出子包，供各问题的结果构建脚本导入。",
    "src/features/__init__.py": "声明特征工程子包，供问题一和问题二脚本导入。",
    "src/models/__init__.py": "声明模型子包，供四问的模型模块导入。",
    "src/utils/__init__.py": "声明工具子包，供一键运行和交付核验模块导入。",
    "scripts/run_paper_mainline.py": "按既定顺序调用 Q1--Q4 的全部主线脚本，并检查交付结果是否齐全。",
    "scripts/build_search_theme_features.py": "由清洗后的百度搜索观测构造主题化搜索特征。",
    "scripts/build_pressure_baseline.py": "拟合问题一的透明相对旅游压力基准模型。",
    "scripts/build_censored_observation_diagnostics.py": "诊断景区排名观测的左删失结构与样本覆盖。",
    "scripts/build_hierarchical_censored_pressure.py": "估计分区分层左删失客流压力，并输出连续日序列。",
    "scripts/build_censored_likelihood_visitor_scale.py": "用外部锚点把相对压力校准为游客规模估计。",
    "scripts/build_q1_timeseries_analysis.py": "生成问题一的时序分解、校验和质量报告。",
    "scripts/build_q1_paper_figures.py": "从问题一结果生成论文图和结果表。",
    "scripts/build_q2_forecasts.py": "完成问题二的双尺度预测与外部锚点校验。",
    "scripts/build_q3_absolute_resource_plan.py": "在游客规模情景下求解停车、接驳和排班配置。",
    "scripts/build_q4_scenario_analysis.py": "模拟雨天和事件冲击并比较重优化方案。",
    "scripts/build_delivery_diagnostics.py": "汇总四问交付状态并写入可复核的完成记录。",
    "src/utils/paper_mainline.py": "定义一键复现时各主线脚本的固定执行顺序。",
    "src/utils/delivery_status.py": "核验各问题的必要输出，并在失败时保留已验证成果。",
    "src/features/search_features.py": "对搜索词时间序列进行质量控制和主题聚合。",
    "src/features/search_keyword_rules.py": "给出搜索词语义分类和可用性筛选规则。",
    "src/features/censored_diagnostics.py": "计算左删失观测的覆盖、阈值和诊断指标。",
    "src/models/pressure_baseline.py": "提供问题一的两状态正则化压力基准模型及滚动检验。",
    "src/models/hierarchical_censored_pressure.py": "构建并拟合结合协变量、空间分区和左删失机制的压力模型。",
    "src/models/scale_calibration.py": "将相对压力与报道锚点结合，形成带不确定性的游客规模。",
    "src/analysis/q1_timeseries.py": "整理问题一的趋势、季节性与分区联动分析输出。",
    "src/analysis/q1_paper_figures.py": "将问题一分析结果绘制为论文使用的图表。",
    "src/analysis/q2_forecast_outputs.py": "生成问题二预测、滚动起点验证和区域搜索滞后选择结果。",
    "src/models/question2_forecasting.py": "定义问题二日级预测所用的特征、模型与评价计算。",
    "src/analysis/q3_absolute_resource_outputs.py": "把绝对游客规模转换为逐日停车、接驳和人员需求。",
    "src/analysis/vot_outputs.py": "将游客时间损失按明确参数换算为可比较的货币成本。",
    "src/models/question3_absolute_optimizer.py": "搜索满足约束的绝对资源配置，并权衡成本和服务损失。",
    "src/models/question3_absolute_capacity.py": "给出停车泊位、接驳车辆和人员数量的承载力计算。",
    "src/models/question3_staff_scheduling.py": "求解三个重叠班次下的最小可行排班。",
    "src/models/value_of_time.py": "定义游客时间价值和时间损失的透明计算公式。",
    "src/analysis/q4_scenario_outputs.py": "将问题四冲击情景接入问题三资源模型并比较方案。",
    "src/models/question4_scenarios.py": "生成连续降雨、文化活动等反事实客流冲击及服务缺口。",
}


def _relative_paths(root: Path, directory: str, category: str) -> list[dict[str, object]]:
    base = root / directory
    if not base.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        rows.append({"relative_path": relative, "category": category, "bytes": path.stat().st_size})
    return rows


def _mainline_source_rows(root: Path) -> list[dict[str, object]]:
    """Return only the explicitly audited source closure for paper reproduction."""
    rows: list[dict[str, object]] = []
    for relative in MAINLINE_SOURCE_FILES:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"mainline source file is missing: {path}")
        rows.append({"relative_path": relative, "category": "source_code", "bytes": path.stat().st_size})
    return rows


def build_support_manifest(root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Return the complete, project-relative contest support-material manifest."""
    root = root.resolve()
    rows: list[dict[str, object]] = []
    for file_name, category in [("requirements.txt", "environment")]:
        path = root / file_name
        if path.exists():
            rows.append({"relative_path": file_name, "category": category, "bytes": path.stat().st_size})
    rows.extend(_mainline_source_rows(root))
    rows.extend(_relative_paths(root, "data/submission", "input_data"))
    rows.extend(_relative_paths(root, "outputs/submission", "result_artifact"))
    for filename in PAPER_FIGURE_FILES:
        path = root / "outputs/figures" / filename
        if not path.is_file():
            raise FileNotFoundError(f"paper figure is missing: {path}")
        rows.append({"relative_path": path.relative_to(root).as_posix(), "category": "result_artifact", "bytes": path.stat().st_size})
    manifest = pd.DataFrame(rows, columns=["relative_path", "category", "bytes"])
    if manifest.empty or manifest["relative_path"].duplicated().any():
        raise ValueError("support-material manifest is empty or contains duplicate paths")
    if manifest["relative_path"].map(lambda value: Path(value).is_absolute() or ".." in Path(value).parts).any():
        raise ValueError("support-material manifest contains an unsafe path")
    return manifest.sort_values(["category", "relative_path"], kind="stable").reset_index(drop=True)


def _tex_path(value: str) -> str:
    return value.replace("_", r"\_")


def _source_group(relative_path: str) -> str:
    lower = relative_path.lower()
    if "common_runtime" in lower:
        return "公共运行工具"
    if "question1" in lower:
        return "问题一：时序构建"
    if "question2" in lower:
        return "问题二：客流预测"
    if "question3" in lower:
        return "问题三：资源配置"
    if "question4" in lower:
        return "问题四：情景重优化"
    if "question3" in lower or "q3_" in lower or "value_of_time" in lower or "vot_" in lower:
        return "问题三：资源配置"
    if "question4" in lower or "q4_" in lower:
        return "问题四：情景重优化"
    if "question2" in lower or "q2_" in lower:
        return "问题二：客流预测"
    if ("hierarchical" in lower or "pressure" in lower or "q1_" in lower or "censored" in lower
            or "search" in lower or "scale_calibration" in lower):
        return "问题一：时序构建"
    return "共同预处理、校验与主线复现"


def write_appendix_fragments(manifest: pd.DataFrame, paper_directory: Path) -> tuple[Path, Path]:
    """Write the compact, paper-facing support-file list; ZIP holds full materials."""
    paper_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = paper_directory / "appendix_support_manifest.tex"
    source_path = paper_directory / "appendix_source_listings.tex"
    manifest_lines = [
        r"\section{支撑文件列表}",
        r"\noindent\fbox{\begin{minipage}{0.93\textwidth}",
        r"\textbf{支撑文件列表：} \texttt{B题支撑材料.zip}（含清洗后数据、结果表与运行环境）\\[-0.35em]",
    ]
    display = manifest.loc[manifest["relative_path"].ne("outputs/figures/.gitkeep")].copy()
    display_groups = [
        ("environment", "运行环境"),
        ("input_data", "清洗后主数据"),
        ("result_artifact", "主结果表"),
        ("source_code", "建模程序"),
    ]
    for category, label in display_groups:
        files = display.loc[display["category"].eq(category), "relative_path"].tolist()
        if category == "result_artifact":
            for prefix, subgroup in [("outputs/submission/", "主结果表"), ("outputs/figures/", "论文图件")]:
                selected = [path for path in files if path.startswith(prefix)]
                if selected:
                    manifest_lines.append(rf"\textbf{{{subgroup}}}\\[-0.45em]")
                    manifest_lines.extend(rf"\texttt{{{_tex_path(Path(path).name)}}}\\[-0.45em]" for path in selected)
            continue
        if files:
            manifest_lines.append(rf"\textbf{{{label}}}\\[-0.45em]")
            if category == "source_code":
                for path in MAINLINE_SOURCE_FILES:
                    if path in files:
                        manifest_lines.append(rf"\texttt{{{_tex_path(Path(path).name)}}}\\[-0.45em]")
            else:
                manifest_lines.extend(rf"\texttt{{{_tex_path(Path(path).name)}}}\\[-0.45em]" for path in files)
    manifest_lines.append(r"\end{minipage}}")
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    source_lines = [
        r"\clearpage",
        r"\noindent\textbf{建模核心程序}\quad 完整运行依赖见支撑材料压缩包；"
        r"以下仅展示四问的核心建模实现。",
    ]
    for relative_path in MAINLINE_SOURCE_FILES:
        if relative_path.endswith("common_runtime.py"):
            continue
        question = _source_group(relative_path).split("：", maxsplit=1)[0]
        source_lines.append(rf"\noindent\textbf{{{question}的程序：}}")
        excerpt_name = Path(relative_path).name.replace("_model.py", "_core.py")
        source_lines.append(rf"\lstinputlisting[style=appendixcode]{{../../../final_programs/appendix_core/{excerpt_name}}}")
    source_path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")
    return manifest_path, source_path


def build_submission_support_package(
    root: Path = PROJECT_ROOT,
    output_directory: Path | None = None,
    paper_directory: Path | None = None,
) -> dict[str, Path]:
    """Write the canonical manifest, TeX fragments, and a <=20 MiB support ZIP."""
    root = root.resolve()
    output = output_directory or root / "outputs/submission_support"
    paper = paper_directory or root / "论文相关模板/国赛Latex模板/国赛模板"
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_support_manifest(root)
    manifest_file = output / "support_materials_manifest.csv"
    manifest.to_csv(manifest_file, index=False, encoding="utf-8-sig")
    manifest_tex, source_tex = write_appendix_fragments(manifest, paper)
    archive = output / "B题支撑材料.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for row in manifest.itertuples(index=False):
            source = root / str(row.relative_path)
            if not source.is_file():
                raise FileNotFoundError(source)
            bundle.write(source, arcname=str(row.relative_path))
        bundle.write(manifest_file, arcname="support_materials_manifest.csv")
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError(f"support ZIP exceeds 20 MiB: {archive.stat().st_size} bytes")
    return {"manifest": manifest_file, "manifest_tex": manifest_tex, "source_tex": source_tex, "archive": archive}


if __name__ == "__main__":
    artifacts = build_submission_support_package()
    for name, path in artifacts.items():
        print(f"{name}: {path}")
