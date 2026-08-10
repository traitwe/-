from pathlib import Path

from src.viz.plots import save_example_figure


def test_save_example_figure_creates_png(tmp_path: Path) -> None:
    output = tmp_path / "example.png"

    save_example_figure(output)

    assert output.is_file()
    assert output.stat().st_size > 0
