import pandas as pd
import pytest

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN
from src.load_data import drop_leakage_columns, load_data, split_features_target


def test_load_data_returns_non_empty_dataframe():
    df = load_data()

    assert isinstance(df, pd.DataFrame)
    assert not df.empty


def test_load_data_contains_target_column():
    df = load_data()

    assert TARGET_COLUMN in df.columns


def test_split_features_target_separates_target():
    df = pd.DataFrame(
        {
            "feature": [1, 2, 3],
            TARGET_COLUMN: [0, 1, 0],
        }
    )

    X, y = split_features_target(df)

    assert TARGET_COLUMN not in X.columns
    assert y.tolist() == [0, 1, 0]


def test_split_features_target_requires_target_column():
    with pytest.raises(ValueError, match=TARGET_COLUMN):
        split_features_target(pd.DataFrame({"feature": [1]}))


def test_drop_leakage_columns_removes_existing_columns_only():
    df = pd.DataFrame(
        {
            "feature": [1],
            "trip_id": ["T1"],
            "actual_departure_delay_min": [5],
            "actual_arrival_delay_min": [7],
        }
    )

    cleaned = drop_leakage_columns(df)

    assert all(column not in cleaned.columns for column in LEAKAGE_COLUMNS)
    assert "feature" in cleaned.columns
