from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.q1_paper_figures import build_q1_paper_figures, build_q1_paper_tables

paths = build_q1_paper_figures(ROOT / "outputs/question1_analysis", ROOT / "outputs/figures")
table_dir = ROOT / "outputs/question1_analysis/tables"
table_dir.mkdir(parents=True, exist_ok=True)
def markdown_table(table):
    rows = ["| " + " | ".join(map(str, table.columns)) + " |", "|" + "|".join(["---"] * len(table.columns)) + "|"]
    rows.extend("| " + " | ".join(map(str, row)) + " |" for row in table.itertuples(index=False, name=None))
    return "\n".join(rows) + "\n"
for name, table in build_q1_paper_tables(ROOT / "outputs/question1_analysis").items():
    table.to_csv(table_dir / f"{name}.csv", index=False, encoding="utf-8-sig")
    (table_dir / f"{name}.md").write_text(markdown_table(table), encoding="utf-8")
print("\n".join(str(path) for path in paths.values()))
