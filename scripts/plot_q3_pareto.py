"""Render the paper-ready Question 3 relative cost--experience tradeoff figure."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = pd.read_csv(ROOT / "outputs/question3_analysis/q3_cost_experience_pareto_2026.csv", encoding="utf-8-sig")
figure, axis = plt.subplots(figsize=(6.6, 4.0))
axis.plot(source["mean_relative_cost"], source["mean_standardized_service_risk"], marker="o", color="#4C78A8")
for _, row in source.iterrows():
    axis.annotate(f"λ={row['risk_penalty']:g}", (row["mean_relative_cost"], row["mean_standardized_service_risk"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
recommended = source.loc[source["risk_penalty"].eq(4.0)].iloc[0]
axis.scatter([recommended["mean_relative_cost"]], [recommended["mean_standardized_service_risk"]], color="#E45756", zorder=3, label="recommended λ=4")
axis.set(xlabel="Mean relative configuration cost", ylabel="Mean standardized service risk", title="Question 3: cost--experience tradeoff")
axis.legend(); figure.tight_layout()
figure.savefig(ROOT / "outputs/question3_analysis/q3_cost_experience_pareto.png", dpi=220)
