from src.utils.paper_mainline import PAPER_MAINLINE_SCRIPTS


def test_paper_mainline_excludes_archived_relative_and_pareto_modules():
    assert "build_q3_absolute_resource_plan.py" in PAPER_MAINLINE_SCRIPTS
    assert "build_q4_scenario_analysis.py" in PAPER_MAINLINE_SCRIPTS
    assert "build_q3_relative_recommendations.py" not in PAPER_MAINLINE_SCRIPTS
    assert "plot_q3_pareto.py" not in PAPER_MAINLINE_SCRIPTS
