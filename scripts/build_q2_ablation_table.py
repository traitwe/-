"""Export one compact ablation: calendar-only versus dynamic daily Ridge."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "outputs" / "question2_analysis" / "q2_daily_model_comparison.csv"
out = ROOT / "outputs" / "question2_analysis" / "q2_ablation_dynamic_covariates_2026.csv"

frame = pd.read_csv(source, encoding="utf-8-sig")
result = frame.loc[frame["candidate"].isin(["seasonal_calendar_ridge", "dynamic_ridge"])].copy()
result["ablation_setting"] = result["candidate"].map({
    "seasonal_calendar_ridge": "remove_weather_and_lagged_search",
    "dynamic_ridge": "retain_weather_and_region_selected_lagged_search",
})
baseline = result.loc[result["candidate"].eq("seasonal_calendar_ridge")].iloc[0]
for metric in ["mae", "rmse", "smape"]:
    result[f"improvement_vs_calendar_{metric}"] = float(baseline[metric]) - result[metric]
result["comparison_scope"] = "same_2024_holdout__same_pressure_to_observed_scale_mapping"
result.to_csv(out, index=False, encoding="utf-8-sig")
