"""Generate the two comparison figures referenced by the paper.

Figure A  -> figures/q3_resource_demand.png  (Table 6: Q3 key resource demand)
Figure B  -> figures/q4_scenario_compare.png (Table 8: Q4 re-optimisation result)

The plotted numbers are the exact values reported in the paper's tables, kept
verbatim so the figures can be independently reproduced from the same figures
script.  Run from this directory:

    python make_paper_figures.py <paper-figures-dir>

<paper-figures-dir> defaults to ./figures and should be the folder the paper
references via \\graphicspath.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _setup_style() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False


def _grouped_bar(fig, ax, tick_labels, series, colors, title, ylabel=None, top=None):
    """Draw a grouped bar chart with value labels on top of each bar."""
    width = 0.36
    x = np.arange(len(tick_labels))
    for offset, (name, values) in enumerate(series.items()):
        pos = x + (offset - 0.5) * width
        bars = ax.bar(pos, values, width, label=name, color=colors[name])
        for rect in bars:
            ax.annotate(
                f"{int(rect.get_height()):,}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel)
    if top is not None:
        ax.set_ylim(0, top)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title(title, fontsize=11)


def fig_q3_resource_demand() -> plt.Figure:
    """Figure A: three-region Q3 resource demand, median vs high-pressure."""
    regions = ["北戴河", "海港--阿那亚", "山海关"]
    # Values verbatim from Table 6 (Q3 key resource demand scenarios).
    parking = {"中位": [9389, 3898, 9980], "高压": [27694, 12245, 47566]}
    shuttle = {"中位": [34, 10, 22],   "高压": [100, 38, 137]}
    staff   = {"中位": [291, 125, 295], "高压": [583, 263, 966]}
    colors = {"中位": "#4C72B0", "高压": "#C44E52"}

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    panels = [
        ("峰值泊位需求（个）", parking, None),
        ("接驳车辆（辆）", shuttle, None),
        ("人员班次（班次）", staff, None),
    ]
    for ax, (title, data, top) in zip(axes, panels):
        _grouped_bar(fig, ax, regions, data, colors, title, top=top)
    fig.suptitle("三片区 Q3 关键资源需求：中位 vs 高压情景", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


def fig_q4_scenario_compare() -> plt.Figure:
    """Figure B: Q4 baseline gap vs post-re-optimisation residual loss."""
    scenarios = ["活动冲击\n(北戴河 8/6--8/10)", "连续降雨\n(山海关 7/15--7/19)"]
    # Values verbatim from Table 8 (counterfactual re-optimisation result).
    baseline_gap = [0.183, 0.259]
    residual_loss = [0.000, 0.099]
    width = 0.32
    x = np.arange(2)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    bars = [
        ax.bar(x - width / 2, baseline_gap, width, label="基线方案最大缺口指数 $G_d$", color="#4C72B0"),
        ax.bar(x + width / 2, residual_loss, width, label="重优化后残余体验损失 $E^{loss}$", color="#55A868"),
    ]
    for group in bars:
        for rect in group:
            ax.annotate(
                f"{rect.get_height():.3f}",
                (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                textcoords="offset points", xytext=(0, 3),
                ha="center", fontsize=9,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("缺口指数（无量纲）")
    ax.set_ylim(0, 0.31)
    ax.legend(fontsize=9, frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.suptitle("Q4 两场景：基线配置在扰动下的不适配 vs 重优化后的残余缺口", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    _setup_style()

    fig_q3_resource_demand().savefig(out_dir / "q3_resource_demand.png", dpi=170)
    fig_q4_scenario_compare().savefig(out_dir / "q4_scenario_compare.png", dpi=170)
    plt.close("all")
    print(f"figures written to {out_dir}")


if __name__ == "__main__":
    main()
