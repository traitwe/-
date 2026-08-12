"""Create delivery-layer diagnostics for Q4 plateaus and POI semantic coverage."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    q4_dir = ROOT / "outputs" / "runtime" / "question4_analysis"
    tradeoff = pd.read_csv(q4_dir / "q4_lambda_tradeoff_2026.csv", encoding="utf-8-sig")
    key = ["mean_relative_cost", "mean_experience_loss", "max_temporary_spaces", "max_shuttle_vehicles", "max_staff_shifts"]
    tradeoff["same_as_previous"] = tradeoff[key].eq(tradeoff[key].shift()).all(axis=1)
    tradeoff["diagnostic"] = tradeoff["same_as_previous"].map({True: "discrete_solution_plateau__not_continuous_marginal_effect", False: "resource_configuration_switch"})
    tradeoff.to_csv(q4_dir / "q4_lambda_plateau_diagnostic_2026.csv", index=False, encoding="utf-8-sig")

    poi = pd.read_csv(ROOT / "data" / "runtime" / "clean" / "raw_cleaned" / "poi_qinhuangdao_purchased.csv", encoding="utf-8-sig")
    rules = {
        "hotel_accommodation": "酒店|宾馆|民宿|旅馆|客栈|度假村",
        "parking": "停车场|停车",
        "attraction_recreation": "景区|公园|海滩|浴场|博物馆|长城|动物园|乐园|广场",
        "transport": "车站|公交|码头|机场|高铁|铁路",
    }
    name = poi["name"].fillna("").astype(str)
    rows = []
    for label, pattern in rules.items():
        matched = name.str.contains(pattern, regex=True)
        rows.append({"semantic_family": label, "keyword_rule": pattern, "poi_count": int(matched.sum()), "share_of_purchased_poi": float(matched.mean()), "interpretation": "name_keyword_proxy_for_spatial_coverage_not_an_operating_capacity"})
    rows.append({"semantic_family": "all_purchased_poi", "keyword_rule": "not_applicable", "poi_count": int(len(poi)), "share_of_purchased_poi": 1.0, "interpretation": "raw purchased POI count"})
    pd.DataFrame(rows).to_csv(ROOT / "outputs" / "runtime" / "question1_analysis" / "q1_poi_semantic_coverage.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
