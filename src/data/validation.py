"""用于赛题数据的轻量质量检查。"""

from collections.abc import Sequence
from typing import Any

import pandas as pd


def validate_dataframe(frame: pd.DataFrame, required_columns: Sequence[str]) -> dict[str, Any]:
    """返回行列数量、重复行、缺失字段和各列空值率。"""
    missing = [column for column in required_columns if column not in frame.columns]
    return {
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "duplicate_row_count": int(frame.duplicated().sum()),
        "missing_required_columns": missing,
        "null_rates": {column: float(rate) for column, rate in frame.isna().mean().items()},
    }
