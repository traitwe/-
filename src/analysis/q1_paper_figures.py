from pathlib import Path

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd


def build_q1_paper_tables(analysis_dir: Path) -> dict[str, pd.DataFrame]:
    quality = pd.read_json(analysis_dir / "q1_analysis_quality_report.json", typ="series")
    table1 = pd.DataFrame([
        ["全市年度国内游客量", "2002—2024", "年度", "千人次", "官方绝对量锚点"],
        ["三大片区旅游压力", "2023—2025", "日级", "相对指数", "时序构建与空间联动"],
        ["搜索、天气与节假日", "2016—2026/2023—2025", "日级", "指数/观测值", "协变量动态层"],
    ], columns=["数据项", "时间跨度", "粒度", "单位", "模型用途"])
    table2 = pd.DataFrame([
        ["Y_y", "全市年度国内游客量", "年度尺度绝对约束"],
        ["F_r,t", "片区日级旅游压力", "相对量；不等同真实日游客数"],
        ["X_t", "搜索、天气、节假日等协变量", "解释短期波动"],
        ["S_t", "隐状态（常态/旺季/冲击）", "刻画状态切换"],
    ], columns=["符号", "定义", "作用与口径"])
    table3 = pd.DataFrame([
        ["年度锚点", "2002—2024 年官方全市国内游客量", "用于年度总量约束"],
        ["日级输出", "2023—2025 年三大片区连续压力序列", "相对压力，不作为真实人数"],
        ["质量报告", str(quality.get("coverage", "见质量报告")), "详细指标见 q1_analysis_quality_report.json"],
    ], columns=["项目", "结果", "解释"])
    return {"table1_data_scope": table1, "table2_variables_parameters": table2, "table3_key_results_validation": table3}


def build_q1_paper_figures(analysis_dir: Path, output_dir: Path) -> dict[str, Path]:
    """Create paper-ready Q1 figures from sealed Q1 analysis outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    annual = pd.read_csv(analysis_dir / "q1_city_annual_trend.csv", encoding="utf-8-sig")
    regional = pd.read_csv(analysis_dir / "q1_regional_pressure_decomposition.csv", encoding="utf-8-sig")
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.bar(annual["year"], annual["official_tourists"], color="#91b7d9", label="官方国内游客量")
    ax.plot(annual["year"], annual["trend_component"], color="#b9413e", marker="o", ms=3, lw=2, label="长期趋势")
    ax.set(xlabel="年份", ylabel="国内游客量（千人次）")
    ax.legend(frameon=False, ncol=2); ax.grid(axis="y", alpha=.25); fig.tight_layout()
    annual_png = output_dir / "q1_city_annual_trend.png"
    fig.savefig(annual_png, dpi=300, bbox_inches="tight"); plt.close(fig)

    panel = regional[(regional["date"] >= "2025-06-15") & (regional["date"] <= "2025-10-10")].copy()
    labels = {"BDH": "北戴河", "HGA": "海港—阿那亚", "SHG": "山海关"}
    colors = {"BDH": "#2f6f9f", "HGA": "#c05a45", "SHG": "#5a9b72"}
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for code, label in labels.items():
        d = panel[panel["region_code"] == code]
        ax.plot(pd.to_datetime(d["date"]), d["pressure_index"], lw=1.35, color=colors[code], label=label)
    ax.axvspan(pd.Timestamp("2025-07-01"), pd.Timestamp("2025-08-31"), color="#f4c95d", alpha=.13, label="暑期")
    ax.axvspan(pd.Timestamp("2025-10-01"), pd.Timestamp("2025-10-07"), color="#df7861", alpha=.13, label="国庆")
    ax.set(xlabel="日期", ylabel="旅游压力指数（相对量）")
    ax.legend(frameon=False, ncol=5, fontsize=8); ax.grid(axis="y", alpha=.25); fig.autofmt_xdate(); fig.tight_layout()
    regional_png = output_dir / "q1_regional_peak_pressure_2025.png"
    fig.savefig(regional_png, dpi=300, bbox_inches="tight"); plt.close(fig)
    return {"annual": annual_png, "regional": regional_png}
