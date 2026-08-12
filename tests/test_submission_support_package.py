from pathlib import Path

from scripts.build_submission_support_package import MAINLINE_SOURCE_FILES, build_support_manifest, write_appendix_fragments


def test_support_manifest_contains_complete_reproducibility_materials():
    root = Path(__file__).resolve().parents[1]

    manifest = build_support_manifest(root)

    paths = set(manifest["relative_path"])
    assert "requirements.txt" in paths
    assert "final_programs/question1_model.py" in paths
    assert "final_programs/question4_model.py" in paths
    assert "final_programs/common_runtime.py" in paths
    assert "data/submission/03_attraction_observation_panel.csv" in paths
    assert "outputs/submission/Q2_validation_summary.csv" in paths
    assert all(not Path(path).is_absolute() for path in paths)
    assert manifest.loc[manifest["relative_path"].str.endswith(".py"), "category"].eq("source_code").all()
    assert not any(path.startswith("scripts/") or path.startswith("src/") for path in paths)
    source_paths = manifest.loc[manifest["category"].eq("source_code"), "relative_path"]
    assert len(source_paths) == 9
    assert all(f"final_programs/appendix_core/question{i}_core.py" in set(source_paths) for i in range(1, 5))
    assert "scipy>=1.13,<2.0" in (root / "requirements.txt").read_text(encoding="utf-8")


def test_compact_appendix_is_one_framed_complete_support_file_list(tmp_path):
    manifest = build_support_manifest(Path(__file__).resolve().parents[1])
    manifest_tex, source_tex = write_appendix_fragments(manifest, tmp_path)
    content = manifest_tex.read_text(encoding="utf-8")
    assert r"\noindent\fbox" in content
    assert r"\begin{minipage}" in content
    assert r"\begin{tabular}" not in content
    assert r"\subsection" not in content
    assert r"01\_tourism\_flow\_anchor\_register.csv" in content
    assert r"08\_data\_source\_quality\_register.csv" in content
    assert r"Q1\_daily\_visitor\_estimates.csv" in content
    assert r"Q4\_robustness\_summary.csv" in content
    assert r"q1\_city\_annual\_trend.png" in content
    assert r"requirements.txt" in content
    assert r"question1\_model.py" in content
    assert r"question4\_model.py" in content
    source_content = source_tex.read_text(encoding="utf-8")
    assert source_content.count(r"\lstinputlisting") == 4
    assert "style=appendixcode" in source_content
    assert "title=" not in source_content
    assert "question1_core.py" in source_content
    assert "question2_core.py" in source_content
    assert "question3_core.py" in source_content
    assert "question4_core.py" in source_content
    assert "appendix_core/question1_core.py" in source_content


def test_appendix_lists_all_audited_model_files_but_shows_only_four_core_models(tmp_path):
    manifest = build_support_manifest(Path(__file__).resolve().parents[1])
    manifest_tex, source_tex = write_appendix_fragments(manifest, tmp_path)
    manifest_content = manifest_tex.read_text(encoding="utf-8")
    source_content = source_tex.read_text(encoding="utf-8")
    assert r"\section{支撑文件列表}" in manifest_content
    expected_file_count = len(manifest.loc[manifest["relative_path"].ne("outputs/figures/.gitkeep")])
    assert manifest_content.count(r"\texttt{") == expected_file_count + 1
    assert source_content.count(r"\lstinputlisting") == 4
    assert r"question1\_model.py" in manifest_content
    assert r"question2\_model.py" in manifest_content
    assert r"question3\_model.py" in manifest_content
    assert r"question4\_model.py" in manifest_content
    assert "build_hierarchical_censored_pressure.py" not in source_content
    assert "build_q1_paper_figures.py" not in source_content
    assert "build_search_theme_features.py" not in source_content
    assert "question1_model.py" not in source_content


def test_support_manifest_contains_every_paper_figure_and_appendix_source():
    root = Path(__file__).resolve().parents[1]
    paths = set(build_support_manifest(root)["relative_path"])

    assert "outputs/figures/q3_resource_demand.png" in paths
    assert "outputs/figures/q4_scenario_compare.png" in paths
    for question in range(1, 5):
        assert f"final_programs/appendix_core/question{question}_core.py" in paths
