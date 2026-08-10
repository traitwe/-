"""Audited semantic and quality rules for the 17-keyword B5c source."""

import pandas as pd


def qinhuangdao_search_keyword_rules() -> pd.DataFrame:
    rows = [
        ("秦皇岛", "A", "destination"),
        ("北戴河", "A", "destination"),
        ("山海关", "A", "destination"),
        ("阿那亚", "A", "coastal_resort"),
        ("秦皇岛旅游", "A", "destination"),
        ("秦皇岛景点", "A", "attraction"),
        ("鸽子窝公园", "A", "attraction"),
        ("秦皇岛天气", "A", "weather"),
        ("北戴河旅游", "B", "destination"),
        ("北戴河景点", "B", "attraction"),
        ("山海关景点", "B", "attraction"),
        ("山海关旅游", "C", "intent"),
        ("山海关门票", "C", "intent"),
        ("北戴河攻略", "C", "intent"),
        ("秦皇岛攻略", "C", "intent"),
        ("老龙头门票", "C", "intent"),
        ("北戴河民宿", "D", "accommodation"),
    ]
    return pd.DataFrame(rows, columns=["keyword", "quality_grade", "theme"])
