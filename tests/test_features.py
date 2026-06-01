import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.preprocessing import add_datetime_features, build_preprocessor


def test_add_datetime_features_extracts_calendar_columns():
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02"],
            "time": ["05:15:00", "18:30:00"],
            "value": [1, 2],
        }
    )

    transformed = add_datetime_features(df)

    assert "date_day_of_week" in transformed.columns
    assert "date_month" in transformed.columns
    assert "time_hour" in transformed.columns
    assert "date" not in transformed.columns
    assert "time" not in transformed.columns


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
