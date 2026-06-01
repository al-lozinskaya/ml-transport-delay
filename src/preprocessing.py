import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LEAKAGE_COLUMNS, TARGET_COLUMN


RAW_TIME_COLUMNS = ["date", "time", "scheduled_departure", "scheduled_arrival"]


def _parse_clock(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(str), format="%H:%M:%S", errors="coerce")


def _duration_minutes(start: pd.Series, end: pd.Series) -> pd.Series:
    start_time = _parse_clock(start)
    end_time = _parse_clock(end)
    duration = (end_time - start_time).dt.total_seconds() / 60
    return duration.where(duration >= 0, duration + 24 * 60)


def _drop_forbidden_training_columns(df: pd.DataFrame) -> pd.DataFrame:
    # trip_id — технический идентификатор, а фактические задержки известны только после поездки.
    # actual_arrival_delay_min напрямую раскрывает delayed, поэтому это утечка данных.
    forbidden = [column for column in [*LEAKAGE_COLUMNS, TARGET_COLUMN] if column in df.columns]
    return df.drop(columns=forbidden)


def _fix_event_attendance(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if {"event_type", "event_attendance_est"}.issubset(result.columns):
        no_event = result["event_type"].astype(str).str.strip().str.lower().eq("none")
        result.loc[no_event, "event_attendance_est"] = 0
    return result


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Создать признаки из даты/времени и удалить сырые временные колонки."""
    result = df.copy()

    if "date" in result.columns:
        parsed_date = pd.to_datetime(result["date"], format="%Y-%m-%d", errors="coerce")
        result["month"] = parsed_date.dt.month
        result["day"] = parsed_date.dt.day
        result["is_weekend"] = parsed_date.dt.dayofweek.isin([5, 6]).astype("Int64")

    clock_column = "time" if "time" in result.columns else "scheduled_departure"
    if clock_column in result.columns:
        parsed_time = _parse_clock(result[clock_column])
        result["hour"] = parsed_time.dt.hour
        result["minute"] = parsed_time.dt.minute

    if {"scheduled_departure", "scheduled_arrival"}.issubset(result.columns):
        result["planned_duration_min"] = _duration_minutes(result["scheduled_departure"], result["scheduled_arrival"])

    return result.drop(columns=[column for column in RAW_TIME_COLUMNS if column in result.columns])


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
    """Подготовить признаки без целевой переменной и leakage-колонок."""
    result = _drop_forbidden_training_columns(df)
    result = _fix_event_attendance(result)
    return add_datetime_features(result)
