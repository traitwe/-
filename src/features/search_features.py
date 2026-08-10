"""Model-ready Baidu search features with explicit quality boundaries."""

from __future__ import annotations

import pandas as pd


_REQUIRED = {"date", "keyword", "search_index", "quality_grade", "theme"}


def build_search_theme_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return annual-standardized, non-weather A/B search-theme factors and lags."""
    missing = _REQUIRED.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["search_index"] = pd.to_numeric(data["search_index"], errors="coerce")
    for column in ["keyword", "quality_grade", "theme"]:
        data[column] = data[column].astype("string").str.strip()
    is_weather = data["theme"].str.lower().eq("weather") | data["keyword"].str.contains("天气", na=False)
    data = data.loc[
        data["date"].notna()
        & data["search_index"].notna()
        & data["quality_grade"].isin(["A", "B"])
        & ~is_weather
    ].copy()
    if data.empty:
        return pd.DataFrame(columns=["date", "source_keyword_count"])

    data["year"] = data["date"].dt.year
    group = data.groupby(["keyword", "year"])["search_index"]
    mean = group.transform("mean")
    std = group.transform("std").fillna(0)
    data["annual_z"] = ((data["search_index"] - mean) / std.where(std.ne(0), 1)).fillna(0)

    theme_daily = (
        data.groupby(["date", "theme"], as_index=False)["annual_z"]
        .mean()
        .pivot(index="date", columns="theme", values="annual_z")
        .rename(columns=lambda theme: f"theme_{theme}")
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )
    counts = data.groupby("date")["keyword"].nunique().rename("source_keyword_count").reset_index()
    result = theme_daily.merge(counts, on="date", how="left")
    theme_columns = [column for column in result.columns if column.startswith("theme_")]
    for column in theme_columns:
        result[f"{column}_lag_1"] = result[column].shift(1)
        result[f"{column}_lag_7"] = result[column].shift(7)
    return result
