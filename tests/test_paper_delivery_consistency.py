from pathlib import Path


PAPER = (
    Path(__file__).resolve().parents[1]
    / "论文相关模板"
    / "国赛Latex模板"
    / "国赛模板"
    / "B题完整论文_国赛模板.tex"
)


def test_paper_is_locked_to_run_b_and_declares_ai_use():
    text = PAPER.read_text(encoding="utf-8")

    assert "日级：季节--日历 Ridge & 0.897 & 1.548 & 63.65" in text
    assert "日级：动态 Ridge & 0.860 & 1.454 & 60.41" in text
    assert "山海关 2.475（26 条）" in text
    assert "山海关 & 0 & 135,069 & 469,364" in text
    assert "山海关--高压 & 64,069 & 184 & 1,300" in text
    assert "设计设计" not in text
    assert "全部回归测试通过" in text
    assert "AI工具使用声明" in text
    assert "详细使用情况见支撑材料《AI工具使用详情.pdf》" in text


def test_paper_does_not_claim_a_likelihood_when_using_map_estimation():
    text = PAPER.read_text(encoding="utf-8")

    assert "MAP 估计" in text
    assert "共享尺度参数" in text
    assert r"\lambda_i" not in text
    assert r"\sigma_i" not in text
