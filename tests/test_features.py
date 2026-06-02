import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN
from src.preprocessing import build_preprocessor, drop_raw_time_columns, prepare_features


def test_drop_raw_time_columns_removes_time_columns_without_creating_features():
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "time": ["05:15:00", "18:30:00"],
            "scheduled_departure": ["05:20:00", "18:35:00"],
            "scheduled_arrival": ["06:05:00", "19:15:00"],
            "value": [1, 2],
        }
    )

    transformed = drop_raw_time_columns(df)

    assert transformed.columns.tolist() == ["value"]
    assert {"date", "time", "scheduled_departure", "scheduled_arrival"}.isdisjoint(transformed.columns)
    assert {"month", "day", "is_weekend", "hour", "minute", "planned_duration_min"}.isdisjoint(transformed.columns)


def test_prepare_features_removes_leakage_target_and_fixes_none_events():
    df = pd.DataFrame(
        {
            "trip_id": ["T1", "T2"],
            "date": ["2023-01-07", "2023-01-09"],
            "time": ["05:15:00", "18:30:00"],
            "scheduled_departure": ["05:20:00", "18:35:00"],
            "scheduled_arrival": ["06:05:00", "19:15:00"],
            "actual_departure_delay_min": [10, 0],
            "actual_arrival_delay_min": [12, 0],
            "event_type": ["None", "Concert"],
            "event_attendance_est": [500, 2500],
            TARGET_COLUMN: [1, 0],
        }
    )

    transformed = prepare_features(df)

    forbidden_columns = set(LEAKAGE_COLUMNS + [TARGET_COLUMN, "date", "time", "scheduled_departure", "scheduled_arrival"])
    assert forbidden_columns.isdisjoint(transformed.columns)
    assert {"month", "day", "is_weekend", "hour", "minute", "planned_duration_min"}.isdisjoint(transformed.columns)
    assert transformed["event_attendance_est"].tolist() == [0, 2500]


def test_build_preprocessor_returns_column_transformer():
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, None, 22.5],
            "transport_type": ["Bus", "Metro", None],
        }
    )

    preprocessor = build_preprocessor(X)

    assert isinstance(preprocessor, ColumnTransformer)


def test_preprocessing_output_has_no_missing_values():
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, None, 22.5],
            "transport_type": ["Bus", "Metro", None],
        }
    )

    preprocessor = build_preprocessor(X)
    transformed = preprocessor.fit_transform(X)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert not np.isnan(dense).any()
