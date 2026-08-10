"""Build model-ready search theme features from cleaned B5c observations."""

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features.search_features import build_search_theme_features
from src.features.search_keyword_rules import qinhuangdao_search_keyword_rules


def main() -> None:
    source = ROOT / "data" / "clean" / "daily_search_heat_17_keywords_2016_2026.csv"
    output_dir = ROOT / "data" / "model_input"
    output_dir.mkdir(parents=True, exist_ok=True)
    heat = pd.read_csv(source, encoding="utf-8-sig")
    rules = qinhuangdao_search_keyword_rules()
    feature_source = heat.merge(rules, on="keyword", how="left", validate="many_to_one")
    features = build_search_theme_features(feature_source)
    rules.to_csv(output_dir / "search_keyword_rules.csv", index=False, encoding="utf-8-sig")
    features.to_csv(output_dir / "daily_search_theme_features_2016_2026.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
