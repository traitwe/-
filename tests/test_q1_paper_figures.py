from pathlib import Path


def test_q1_paper_figure_builder_writes_two_pngs(tmp_path):
    from src.analysis.q1_paper_figures import build_q1_paper_figures

    root = Path(__file__).resolve().parents[1]
    created = build_q1_paper_figures(root / "outputs/question1_analysis", tmp_path)

    assert set(created) == {"annual", "regional"}
    assert all(path.exists() and path.stat().st_size > 10_000 for path in created.values())
