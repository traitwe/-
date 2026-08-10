"""最小化的竞赛图表输出工具。"""

from pathlib import Path

import matplotlib.pyplot as plt


def save_example_figure(output_path: Path) -> None:
    """输出一张可替换的示例图，验证图表目录和保存流程。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.plot([1, 2, 3], [1, 4, 9], marker="o")
    axis.set(xlabel="x", ylabel="y", title="Example competition figure")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
