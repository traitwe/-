import pandas as pd

from src.data.validation import validate_dataframe


def test_validation_counts_rows_duplicates_and_missing_columns() -> None:
    frame = pd.DataFrame({"x": [1, 1], "y": [None, 2]})

    report = validate_dataframe(frame, required_columns=["x", "z"])

    assert report["row_count"] == 2
    assert report["duplicate_row_count"] == 0
    assert report["missing_required_columns"] == ["z"]
    assert report["null_rates"]["y"] == 0.5
