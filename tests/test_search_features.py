import pandas as pd

from src.features.search_features import build_search_theme_features


def test_search_features_excludes_weather_and_standardizes_within_year() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2025-01-01", "2025-01-02"]),
            "keyword": ["北戴河", "北戴河", "北戴河", "北戴河"],
            "search_index": [10, 20, 100, 120],
            "quality_grade": ["A", "A", "A", "A"],
            "theme": ["destination", "destination", "destination", "destination"],
        }
    )
    weather = frame.assign(keyword="秦皇岛天气", search_index=[1000, 2000, 3000, 4000], theme="weather")

    result = build_search_theme_features(pd.concat([frame, weather], ignore_index=True))

    assert "theme_destination_lag_1" in result.columns
    assert not any("weather" in column for column in result.columns)
    assert result.loc[result["date"].dt.year.eq(2024), "theme_destination"].mean() == 0
    assert result.loc[result["date"].dt.year.eq(2025), "theme_destination"].mean() == 0


def test_search_features_uses_only_a_and_b_keywords_for_theme_factors() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01"] * 4),
            "keyword": ["北戴河", "山海关", "山海关攻略", "北戴河民宿"],
            "search_index": [10, 20, 999, 888],
            "quality_grade": ["A", "B", "C", "D"],
            "theme": ["destination", "destination", "intent", "accommodation"],
        }
    )

    result = build_search_theme_features(frame)

    assert result.loc[0, "source_keyword_count"] == 2
    assert "theme_intent" not in result.columns
    assert "theme_accommodation" not in result.columns
