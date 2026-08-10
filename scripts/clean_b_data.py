"""Create clean B题 inputs from immutable local raw sources and a friend archive.

Usage:
    python scripts/clean_b_data.py --friend-zip "C:\\Users\\lyy20\\Downloads\\数据 - 副本.zip"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.cleaning import (
    clean_attraction_area,
    clean_capacity_standard,
    clean_poi,
    clean_regional_area_scenarios,
    clean_reservation_scenarios,
    clean_search_heat,
    clean_service_time_evidence,
    clean_tabular_source,
    profile_dataframe,
)


RAW_DIR = PROJECT_ROOT / "data" / "raw"
CLEAN_DIR = PROJECT_ROOT / "data" / "clean"
READ_NOTES: dict[str, str] = {}


FRIEND_FILES = {
    "annual_tourism": "数据 - 副本/L0_原始数据/A_客流/L0_A1b_全市旅游面板_年度_1999_2024_用户提供.csv",
    "flow_index": "数据 - 副本/L0_原始数据/A_客流/L0_A3b_景区拥堵与客流指数_秦皇岛_日_2021_2025_用户提供.csv",
    "weather": "数据 - 副本/L0_原始数据/B_影响因子/L0_B1d_气象日度_2015_2025_脚本抓取.csv",
    "calendar": "数据 - 副本/L0_原始数据/B_影响因子/L0_B2b_日历与调休_2015_2026_脚本抓取.csv",
    "search_heat": "数据 - 副本/L0_原始数据/B_影响因子/L0_B5b_网络热度_百度指数_日_2016_2026_用户提供.csv",
    "attraction_poi": "数据 - 副本/L0_原始数据/A_客流/L0_A8_旅游景点POI_秦皇岛_用户提供.csv",
    "parking_poi": "数据 - 副本/L0_原始数据/C_资源运营/L0_C1b_停车场POI_秦皇岛_用户提供.csv",
    "hotel_poi": "数据 - 副本/L0_原始数据/C_资源运营/L0_C9_酒店住宿POI_秦皇岛_用户提供.csv",
    "transit_poi": "数据 - 副本/L0_原始数据/C_资源运营/L0_C10_公交站POI_秦皇岛_用户提供.csv",
}


def _read_friend_csv(archive: zipfile.ZipFile, member: str) -> pd.DataFrame:
    with archive.open(member) as stream:
        return pd.read_csv(io.TextIOWrapper(stream, encoding="utf-8-sig"), low_memory=False)


def _read_project_csv(path: Path) -> pd.DataFrame:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            continue
        except pd.errors.ParserError:
            READ_NOTES[str(path.relative_to(PROJECT_ROOT))] = (
                "Malformed CSV record(s): parsed with Python engine and malformed rows excluded; "
                "inspect the original source before using excluded observations."
            )
            return pd.read_csv(path, encoding=encoding, engine="python", on_bad_lines="warn")
    raise ValueError(f"cannot decode CSV: {path}")


def _clean_flow_index(frame: pd.DataFrame) -> pd.DataFrame:
    required_mapping = {
        "景区名称": "attraction_name",
        "片区代码": "region_code",
        "区县": "district",
        "周边路网拥堵指数": "road_congestion_index",
        "拥堵指数较平日": "congestion_vs_normal",
        "客流指数": "visitor_index",
        "客流指数较平日": "visitor_vs_normal",
        "日期": "date",
        "当日排名": "daily_rank",
    }
    optional_mapping = {
        "标签": "tags",
        "经度": "longitude",
        "纬度": "latitude",
        "拥堵指数异常": "congestion_anomaly",
        "省份": "province",
        "城市": "city",
    }
    missing = [column for column in required_mapping if column not in frame.columns]
    if missing:
        raise ValueError(f"flow-index data missing required columns: {missing}")
    mapping = required_mapping | {column: name for column, name in optional_mapping.items() if column in frame.columns}
    result = frame.loc[:, list(mapping)].rename(columns=mapping)
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    for column in ["road_congestion_index", "congestion_vs_normal", "visitor_index", "visitor_vs_normal", "daily_rank"]:
        result[column] = pd.to_numeric(result[column].replace("-", pd.NA), errors="coerce")
    for column in ["longitude", "latitude"]:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    result["attraction_name"] = result["attraction_name"].astype("string").str.strip()
    result = result.dropna(subset=["date", "attraction_name", "visitor_index"])
    result = result.drop_duplicates(subset=["date", "attraction_name"], keep="first")
    result = result.sort_values(["attraction_name", "date"]).reset_index(drop=True)
    result["data_type"] = "relative_index_not_absolute_visitors"
    result["source_label"] = "friend_archive_baidu_map"
    return result


def _clean_daily_weather(frame: pd.DataFrame) -> pd.DataFrame:
    date_candidates = [column for column in frame.columns if column in {"日期", "date"}]
    if not date_candidates:
        raise ValueError("weather data has no date column")
    date_column = date_candidates[0]
    result = frame.copy()
    result = result.rename(columns={date_column: "date"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)
    result["source_label"] = "friend_archive_open_meteo_era5"
    return result


def _clean_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    date_column = "日期" if "日期" in frame.columns else "date"
    result = frame.copy().rename(columns={date_column: "date"})
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="first")
    result = result.sort_values("date").reset_index(drop=True)
    result["source_label"] = "friend_archive_calendar"
    return result


def _clean_annual_tourism(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    year_column = "年份" if "年份" in result.columns else "year"
    result = result.rename(columns={year_column: "year"})
    result["year"] = pd.to_numeric(result["year"], errors="coerce").astype("Int64")
    result = result.dropna(subset=["year"]).drop_duplicates(subset=["year"], keep="first")
    result = result.sort_values("year").reset_index(drop=True)
    result["source_label"] = "friend_archive_city_statistics"
    return result


def _write(frame: pd.DataFrame, filename: str, report: dict[str, object], key_columns: list[str], date_column: str | None) -> None:
    path = CLEAN_DIR / filename
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    report[filename] = profile_dataframe(frame, key_columns=key_columns, date_column=date_column)


def _raw_file(prefix: str) -> Path:
    matches = sorted(RAW_DIR.glob(f"{prefix}*.csv"))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one raw CSV beginning with {prefix!r}, found {len(matches)}")
    return matches[0]


def main(friend_zip: Path) -> None:
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {
        "cleaning_policy": "no imputation; raw sources remain unchanged",
        "tables": {},
        "read_exceptions": READ_NOTES,
    }
    tables: dict[str, object] = report["tables"]  # type: ignore[assignment]

    with zipfile.ZipFile(friend_zip) as archive:
        search_heat = clean_search_heat(_read_friend_csv(archive, FRIEND_FILES["search_heat"]), "friend_archive_baidu_index")
        _write(search_heat, "daily_search_heat_2016_2026.csv", tables, ["date", "keyword"], "date")

        flow = _clean_flow_index(_read_friend_csv(archive, FRIEND_FILES["flow_index"]))
        _write(flow, "daily_attraction_flow_index_2021_2025.csv", tables, ["date", "attraction_name"], "date")

        weather = _clean_daily_weather(_read_friend_csv(archive, FRIEND_FILES["weather"]))
        _write(weather, "daily_weather_2015_2025.csv", tables, ["date"], "date")

        calendar = _clean_calendar(_read_friend_csv(archive, FRIEND_FILES["calendar"]))
        _write(calendar, "calendar_2015_2026.csv", tables, ["date"], "date")

        annual = _clean_annual_tourism(_read_friend_csv(archive, FRIEND_FILES["annual_tourism"]))
        _write(annual, "annual_city_tourism_1999_2024.csv", tables, ["year"], None)

        for family in ["attraction_poi", "parking_poi", "hotel_poi", "transit_poi"]:
            poi = clean_poi(_read_friend_csv(archive, FRIEND_FILES[family]), family.replace("_poi", ""))
            _write(poi, f"poi_{family.replace('_poi', '')}.csv", tables, ["name", "longitude", "latitude"], None)

    heat_17 = clean_search_heat(_read_project_csv(_raw_file("L0_B5c_")), "user_baidu_index_17_keywords")
    _write(heat_17, "daily_search_heat_17_keywords_2016_2026.csv", tables, ["date", "keyword"], "date")

    capacity_standard = clean_capacity_standard(_read_project_csv(_raw_file("L0_C20_")))
    _write(capacity_standard, "capacity_space_standard.csv", tables, ["per_capita_space_baseline_m2"], None)

    attraction_area = clean_attraction_area(_read_project_csv(_raw_file("L0_C21_景区")))
    _write(attraction_area, "attraction_effective_area.csv", tables, ["area_evidence_id"], None)

    regional_area = clean_regional_area_scenarios(_read_project_csv(_raw_file("L0_C21c_")))
    _write(regional_area, "regional_effective_area_scenarios.csv", tables, ["region_code"], None)

    reservation = clean_reservation_scenarios(_read_project_csv(_raw_file("L0_C22_")))
    _write(reservation, "reservation_ratio_scenarios.csv", tables, ["reservation_ratio"], None)

    service_time = clean_service_time_evidence(_read_project_csv(_raw_file("qinhuangdao_hourly_entry_time_evidence_")))
    _write(service_time, "hourly_service_time_evidence.csv", tables, ["time_start", "time_end"], None)

    raw_inventory = []
    for path in sorted(RAW_DIR.rglob("*")):
        if path.is_file():
            raw_inventory.append({"path": str(path.relative_to(PROJECT_ROOT)), "size_bytes": path.stat().st_size, "suffix": path.suffix.lower()})
    inventory = pd.DataFrame(raw_inventory)
    inventory.to_csv(CLEAN_DIR / "raw_inventory.csv", index=False, encoding="utf-8-sig")
    tables["raw_inventory.csv"] = profile_dataframe(inventory, key_columns=["path"], date_column=None)

    raw_clean_dir = CLEAN_DIR / "raw_cleaned"
    raw_clean_dir.mkdir(exist_ok=True)
    for path in sorted(RAW_DIR.glob("*.csv")):
        if path.name == ".gitkeep":
            continue
        if path.name == "秦皇岛市POI.csv":
            cleaned = clean_poi(_read_project_csv(path), "purchased_all_poi")
            output_name = "poi_qinhuangdao_purchased.csv"
            key_columns = ["name", "longitude", "latitude"]
            date_column = None
        else:
            cleaned = clean_tabular_source(_read_project_csv(path))
            output_name = path.name
            key_columns = list(cleaned.columns[:1])
            date_candidates = [column for column in cleaned.columns if str(column).lower().replace(" ", "") in {"date", "日期"}]
            date_column = date_candidates[0] if date_candidates else None
        output_path = raw_clean_dir / output_name
        cleaned.to_csv(output_path, index=False, encoding="utf-8-sig")
        tables[f"raw_cleaned/{output_name}"] = profile_dataframe(cleaned, key_columns=key_columns, date_column=date_column)

    (CLEAN_DIR / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (CLEAN_DIR / "README.md").write_text(
        "# Clean B题 Data\n\n"
        "These outputs are normalized copies of source observations. No missing value is imputed, and all visitor-index data remains labelled as a relative index rather than absolute visitor counts.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--friend-zip", type=Path, required=True)
    args = parser.parse_args()
    main(args.friend_zip)
