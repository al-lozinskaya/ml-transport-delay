import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN


RAW_TIME_COLUMNS = ["date", "time", "scheduled_departure", "scheduled_arrival"]


def _drop_forbidden_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    # trip_id — технический идентификатор.
    # actual_* delay колонки известны после поездки и дают leakage.
    forbidden = [column for column in [*LEAKAGE_COLUMNS, TARGET_COLUMN] if column in df.columns]
    return df.drop(columns=forbidden)


def _fix_event_attendance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"event_type", "event_attendance_est"}.issubset(result.columns):
        no_event = result["event_type"].astype(str).str.strip().str.lower().eq("none")
        result.loc[no_event, "event_attendance_est"] = 0
    return result


def drop_raw_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удалить сырые временные колонки."""
    return df.drop(columns=[column for column in RAW_TIME_COLUMNS if column in df.columns])


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Собрать preprocessing для числовых и категориальных признаков."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Подготовить признаки без target, leakage и сырых временных колонок."""
    result = _drop_forbidden_training_columns(df)
    result = _fix_event_attendance(result)
    return drop_raw_time_columns(result)
