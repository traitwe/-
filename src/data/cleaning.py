"""Reproducible normalization helpers for B题 tourism data.

Raw files are never modified.  Every cleaner returns a new DataFrame with
typed fields and explicit provenance so that modeling code can distinguish
observations from source families.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd


def _require_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def clean_search_heat(frame: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """Normalize daily Baidu-index observations without imputing any value."""
    columns = ["日期", "关键词", "整体指数", "PC端指数", "移动端指数"]
    _require_columns(frame, columns)
    result = frame.loc[:, columns].rename(
        columns={
            "日期": "date",
            "关键词": "keyword",
            "整体指数": "search_index",
            "PC端指数": "pc_index",
            "移动端指数": "mobile_index",
        }
    )
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["search_index", "pc_index", "mobile_index"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    result["keyword"] = result["keyword"].astype("string").str.strip()
    result = result.dropna(subset=["date", "keyword", "search_index"])
    result = result.drop_duplicates(subset=["date", "keyword"], keep="first")
    result = result.sort_values(["keyword", "date"]).reset_index(drop=True)
    result["is_weekend"] = (result["date"].dt.dayofweek >= 5).astype("int8")
    result["source_label"] = source_label
    return result


def clean_poi(frame: pd.DataFrame, poi_family: str) -> pd.DataFrame:
    """Normalize POI locations and retain only valid WGS84-like coordinates."""
    columns = ["名称", "大类", "中类", "经度", "纬度"]
    _require_columns(frame, columns)
    result = frame.loc[:, columns].rename(
        columns={"名称": "name", "大类": "major_category", "中类": "minor_category", "经度": "longitude", "纬度": "latitude"}
    )
    result["name"] = result["name"].astype("string").str.strip()
    result["major_category"] = result["major_category"].astype("string").str.strip()
    result["minor_category"] = result["minor_category"].astype("string").str.strip()
    result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
    result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
    result = result.dropna(subset=["name", "longitude", "latitude"])
    result = result.loc[result["longitude"].between(73, 136) & result["latitude"].between(18, 54)]
    result = result.drop_duplicates(subset=["name", "longitude", "latitude"], keep="first")
    result = result.reset_index(drop=True)
    result["poi_family"] = poi_family
    return result


def clean_tabular_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply conservative cleaning to an anchor table without altering its meaning."""
    result = frame.copy()
    result.columns = [str(column).strip() for column in result.columns]
    for column in result.select_dtypes(include=["object", "string"]).columns:
        result[column] = result[column].astype("string").str.strip()
    for column in result.columns:
        normalized = column.lower().replace(" ", "")
        if normalized in {"date", "日期"} or normalized.endswith("日期"):
            result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.drop_duplicates(keep="first")
    return result.reset_index(drop=True)


def _first_present(frame: pd.DataFrame, candidates: Sequence[str], label: str) -> str:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    raise ValueError(f"{label} has none of the expected columns: {list(candidates)}")


def _parse_space_range(value: object) -> tuple[float | None, float | None]:
    """Extract one or two numeric values from a capacity standard cell."""
    text = str(value).replace("–", "-").replace("—", "-")
    numbers = pd.Series([text]).str.extractall(r"(\d+(?:\.\d+)?)")[0].tolist()
    if not numbers:
        return None, None
    lower = float(numbers[0])
    upper = float(numbers[1]) if len(numbers) > 1 else lower
    return min(lower, upper), max(lower, upper)


def clean_capacity_standard(frame: pd.DataFrame) -> pd.DataFrame:
    """Turn human-readable space-per-visitor standards into numeric scenarios."""
    indicator = _first_present(frame, ["人均空间承载指标", "per_capita_space_standard"], "capacity standard")
    result = frame.copy()
    parsed = result[indicator].map(_parse_space_range)
    result["per_capita_space_lower_m2"] = parsed.map(lambda item: item[0])
    result["per_capita_space_upper_m2"] = parsed.map(lambda item: item[1])
    result["per_capita_space_baseline_m2"] = (
        result["per_capita_space_lower_m2"] + result["per_capita_space_upper_m2"]
    ) / 2
    result = result.dropna(subset=["per_capita_space_baseline_m2"]).reset_index(drop=True)
    return result


def clean_attraction_area(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize area evidence while retaining conflicting sources as separate records."""
    region = _first_present(frame, ["片区代码", "片区", "region_code"], "attraction area")
    attraction = _first_present(frame, ["景区点位", "景区名称", "点位名称", "attraction_name"], "attraction area")
    area = _first_present(frame, ["换算面积_m2", "面积_平方米", "converted_area_m2", "面积_m2"], "attraction area")
    usable = _first_present(frame, ["建模可用", "model_usable"], "attraction area")
    result = frame.copy().rename(
        columns={region: "region_code", attraction: "attraction_name", area: "area_m2", usable: "model_usable_raw"}
    )
    result["region_code"] = result["region_code"].astype("string").str.strip()
    result["attraction_name"] = result["attraction_name"].astype("string").str.strip()
    result["area_m2"] = pd.to_numeric(result["area_m2"], errors="coerce")
    usable_text = result["model_usable_raw"].astype("string").str.strip()
    result["is_model_usable"] = usable_text.str.contains("是|可直接", regex=True, na=False).astype("int8")
    result = result.dropna(subset=["region_code", "attraction_name", "area_m2"]).reset_index(drop=True)
    result.insert(0, "area_evidence_id", [f"area_{index + 1:03d}" for index in range(len(result))])
    return result


def clean_regional_area_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """Standardize supplied S1 effective-area scenarios in square metres."""
    region = _first_present(frame, ["片区代码", "片区", "region_code"], "regional area scenario")
    region_name = next((column for column in ["片区名", "region_name"] if column in frame.columns), None)
    required = {
        "S1_下限_km2": "effective_area_lower_m2",
        "S1_基准_km2": "effective_area_baseline_m2",
        "S1_上限_km2": "effective_area_upper_m2",
        "S0_地理边界_km2": "geographic_area_m2",
    }
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"regional area scenario missing required columns: {missing}")
    result = frame.copy().rename(columns={region: "region_code"})
    result.insert(0, "source_region_code", result["region_code"].astype("string").str.strip())
    result["region_code"] = result["source_region_code"]
    if region_name:
        aranya = result[region_name].astype("string").str.contains("\u963f\u90a3\u4e9a", na=False)
        result.loc[aranya, "region_code"] = "ARA"
    for source, target in required.items():
        result[target] = (pd.to_numeric(result[source], errors="coerce") * 1_000_000).round().astype("Int64")
    result["region_code"] = result["region_code"].astype("string").str.strip()
    result = result.dropna(subset=["region_code", "effective_area_baseline_m2"])
    if result.duplicated(subset=["region_code"]).any():
        raise ValueError("regional area scenario has duplicate model region_code after normalization")
    return result.reset_index(drop=True)


def clean_reservation_scenarios(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep only explicitly usable actual-reservation ratios; retain no proxy opening rates."""
    pct = _first_present(frame, ["数值_pct", "value_pct"], "reservation ratio")
    usable = _first_present(frame, ["可直接用作online_ratio", "use_as_online_ratio"], "reservation ratio")
    result = frame.copy()
    result["reservation_ratio"] = pd.to_numeric(result[pct], errors="coerce") / 100
    usable_text = result[usable].astype("string").str.strip()
    result["is_direct_actual_ratio"] = usable_text.str.fullmatch("是|yes|true", case=False, na=False).astype("int8")
    result = result.loc[result["is_direct_actual_ratio"].eq(1)].dropna(subset=["reservation_ratio"])
    result = result.loc[result["reservation_ratio"].between(0, 1)].reset_index(drop=True)
    return result


def clean_service_time_evidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize service-time evidence and explicitly label it as non-count evidence."""
    start = _first_present(frame, ["开始时间", "时段开始", "time_start"], "service-time evidence")
    end = _first_present(frame, ["结束时间", "时段结束", "time_end"], "service-time evidence")
    result = frame.copy().rename(columns={start: "time_start", end: "time_end"})
    for column in ["time_start", "time_end"]:
        result[column] = result[column].astype("string").str.strip()
        valid = result[column].str.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", na=False)
        result.loc[~valid, column] = pd.NA
    result = result.dropna(subset=["time_start", "time_end"]).drop_duplicates(keep="first").reset_index(drop=True)
    result["is_hourly_count"] = 0
    result["evidence_type"] = "service_time_or_qualitative_peak_not_hourly_count"
    return result


def profile_dataframe(
    frame: pd.DataFrame,
    key_columns: Sequence[str] | None = None,
    date_column: str | None = None,
) -> dict[str, Any]:
    """Return compact quality metadata without changing the input frame."""
    report: dict[str, Any] = {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "null_counts": {column: int(count) for column, count in frame.isna().sum().items()},
    }
    if key_columns:
        _require_columns(frame, key_columns)
        report["duplicate_key_count"] = int(frame.duplicated(subset=list(key_columns)).sum())
    if date_column:
        _require_columns(frame, [date_column])
        parsed = pd.to_datetime(frame[date_column], errors="coerce").dropna().dt.normalize()
        report["date_min"] = parsed.min().strftime("%Y-%m-%d") if not parsed.empty else None
        report["date_max"] = parsed.max().strftime("%Y-%m-%d") if not parsed.empty else None
        report["invalid_date_count"] = int(len(frame) - len(parsed))
        unique_dates = parsed.drop_duplicates()
        report["date_unique_count"] = int(len(unique_dates))
        if parsed.empty:
            report["missing_calendar_days_between_dates"] = None
        else:
            expected = (parsed.max() - parsed.min()).days + 1
            report["missing_calendar_days_between_dates"] = int(expected - len(unique_dates))
    return report
