import pandas as pd

from scripts.clean_b_data import _clean_flow_index
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


def test_clean_search_heat_drops_duplicate_rows_and_adds_calendar_fields() -> None:
    frame = pd.DataFrame(
        {
            "日期": ["2025-07-01", "2025-07-01", "2025-07-02"],
            "关键词": ["北戴河", "北戴河", "北戴河"],
            "整体指数": [100, 100, 120],
            "PC端指数": [40, 40, 50],
            "移动端指数": [60, 60, 70],
        }
    )

    result = clean_search_heat(frame, source_label="friend_baidu_index")

    assert len(result) == 2
    assert result["date"].dtype.kind == "M"
    assert result.loc[0, "is_weekend"] == 0
    assert result.loc[0, "source_label"] == "friend_baidu_index"


def test_clean_poi_discards_invalid_coordinates_and_exact_duplicates() -> None:
    frame = pd.DataFrame(
        {
            "名称": ["A酒店", "A酒店", "坏坐标"],
            "大类": ["酒店住宿", "酒店住宿", "酒店住宿"],
            "中类": ["舒适型酒店", "舒适型酒店", "其他"],
            "经度": [119.6, 119.6, 300.0],
            "纬度": [39.9, 39.9, 39.9],
        }
    )

    result = clean_poi(frame, poi_family="hotel")

    assert len(result) == 1
    assert result.loc[0, "poi_family"] == "hotel"
    assert result.loc[0, "name"] == "A酒店"


def test_profile_dataframe_reports_duplicate_key_count_and_date_range() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-07-01", "2025-07-01"]),
            "keyword": ["北戴河", "北戴河"],
        }
    )

    report = profile_dataframe(frame, key_columns=["date", "keyword"], date_column="date")

    assert report["row_count"] == 2
    assert report["duplicate_key_count"] == 1
    assert report["date_min"] == "2025-07-01"


def test_clean_flow_index_accepts_friend_archive_schema() -> None:
    frame = pd.DataFrame(
        {
            "景区名称": ["鸽子窝公园", "鸽子窝公园"],
            "片区代码": ["BDH", "BDH"],
            "区县": ["北戴河区", "北戴河区"],
            "周边路网拥堵指数": [1.2, 1.2],
            "拥堵指数较平日": [0.1, 0.1],
            "客流指数": [0.8, 0.8],
            "客流指数较平日": [0.2, 0.2],
            "日期": ["2024-07-01", "2024-07-01"],
            "当日排名": [3, 3],
        }
    )

    result = _clean_flow_index(frame)

    assert len(result) == 1
    assert result.loc[0, "region_code"] == "BDH"
    assert result.loc[0, "data_type"] == "relative_index_not_absolute_visitors"


def test_clean_tabular_source_strips_text_normalizes_date_and_removes_exact_rows() -> None:
    frame = pd.DataFrame(
        {
            " 日期 ": ["2024/07/01", "2024/07/01", "2024/07/02"],
            " 名称 ": [" 北戴河 ", " 北戴河 ", " 阿那亚 "],
            "值": [1, 1, 2],
        }
    )

    result = clean_tabular_source(frame)

    assert list(result.columns) == ["日期", "名称", "值"]
    assert len(result) == 2
    assert str(result.iloc[0]["日期"].date()) == "2024-07-01"
    assert result.iloc[0]["名称"] == "北戴河"


def test_profile_dataframe_reports_missing_days_within_observed_date_span() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2025-07-01", "2025-07-03"])})

    report = profile_dataframe(frame, key_columns=["date"], date_column="date")

    assert report["date_unique_count"] == 2
    assert report["missing_calendar_days_between_dates"] == 1


def test_clean_capacity_standard_parses_space_ranges() -> None:
    frame = pd.DataFrame(
        {
            "人均空间承载指标": ["1–1.1 m²/person", "0.5 m²/person"],
            "景区类型": ["海滨", "博物馆"],
        }
    )

    result = clean_capacity_standard(frame)

    assert result.loc[0, "per_capita_space_lower_m2"] == 1.0
    assert result.loc[0, "per_capita_space_upper_m2"] == 1.1
    assert result.loc[0, "per_capita_space_baseline_m2"] == 1.05
    assert result.loc[1, "per_capita_space_baseline_m2"] == 0.5


def test_clean_attraction_area_preserves_conflicting_evidence() -> None:
    frame = pd.DataFrame(
        {
            "片区代码": ["BDH", "BDH"],
            "景区点位": ["鸽子窝", "鸽子窝"],
            "换算面积_m2": [1000, 2000],
            "建模可用": ["是", "否"],
            "来源链接": ["source-a", "source-b"],
        }
    )

    result = clean_attraction_area(frame)

    assert len(result) == 2
    assert result["area_evidence_id"].is_unique
    assert result["is_model_usable"].tolist() == [1, 0]


def test_clean_regional_area_scenarios_converts_km2_to_m2() -> None:
    frame = pd.DataFrame(
        {
            "片区代码": ["BDH"],
            "S1_下限_km2": [3.8],
            "S1_基准_km2": [4.15],
            "S1_上限_km2": [4.5],
            "S0_地理边界_km2": [22.7],
            "kappa_S0toS1": [0.1828],
        }
    )

    result = clean_regional_area_scenarios(frame)

    assert result.loc[0, "effective_area_baseline_m2"] == 4_150_000
    assert result.loc[0, "geographic_area_m2"] == 22_700_000


def test_clean_regional_area_scenarios_separates_aranya_from_haigang() -> None:
    frame = pd.DataFrame(
        {
            "片区": ["HGA", "HGA"],
            "片区名": ["海港区", "阿那亚休闲片区"],
            "S1_下限_km2": [1.5, 0.00138],
            "S1_基准_km2": [1.65, 0.00138],
            "S1_上限_km2": [1.8, 0.00138],
            "S0_地理边界_km2": [701, 10.48],
        }
    )

    result = clean_regional_area_scenarios(frame)

    assert result["region_code"].tolist() == ["HGA", "ARA"]
    assert result["source_region_code"].tolist() == ["HGA", "HGA"]


def test_clean_reservation_scenarios_keeps_only_direct_actual_ratios() -> None:
    frame = pd.DataFrame(
        {
            "数值_pct": [94, 41.4],
            "可直接用作online_ratio": ["否", "是"],
            "口径": ["预约开放率", "实际预约率"],
            "来源": ["A", "B"],
        }
    )

    result = clean_reservation_scenarios(frame)

    assert len(result) == 1
    assert result.loc[0, "reservation_ratio"] == 0.414
    assert result.loc[0, "is_direct_actual_ratio"] == 1


def test_clean_service_time_evidence_marks_non_count_evidence() -> None:
    frame = pd.DataFrame(
        {
            "片区代码": ["BDH", "BDH"],
            "设施": ["旅游巴士", "旅游巴士"],
            "开始时间": ["07:30", "07:30"],
            "结束时间": ["19:30", "19:30"],
            "数据状态": ["时段证据", "时段证据"],
        }
    )

    result = clean_service_time_evidence(frame)

    assert len(result) == 1
    assert result.loc[0, "time_start"] == "07:30"
    assert result.loc[0, "is_hourly_count"] == 0
