from pathlib import Path


def test_final_program_builder_writes_four_entrypoints_and_one_shared_runtime(tmp_path):
    from scripts.build_final_programs import build_final_programs

    root = Path(__file__).resolve().parents[1]
    paths = build_final_programs(root, output_directory=tmp_path)

    assert [path.name for path in paths] == [
        "common_runtime.py", "question1_model.py", "question2_model.py", "question3_model.py", "question4_model.py",
    ]
    common = (tmp_path / "common_runtime.py").read_text(encoding="utf-8")
    assert "prepare_submission_runtime" in common
    assert "_export_submission" in common
    for path in paths[1:]:
        content = path.read_text(encoding="utf-8")
        assert "from src." not in content
        assert "from common_runtime import" in content
        assert "prepare_submission_runtime" in content


def test_final_program_builder_writes_four_short_appendix_snippets(tmp_path):
    from scripts.build_final_programs import build_final_programs

    root = Path(__file__).resolve().parents[1]
    build_final_programs(root, output_directory=tmp_path)
    snippets = sorted((tmp_path / "appendix_core").glob("question*_core.py"))

    assert [path.name for path in snippets] == [
        "question1_core.py", "question2_core.py", "question3_core.py", "question4_core.py",
    ]
    assert all(10 <= len(path.read_text(encoding="utf-8").splitlines()) <= 150 for path in snippets)
