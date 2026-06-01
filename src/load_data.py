from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, LEAKAGE_COLUMNS, TARGET_COLUMN


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """Загрузить CSV-датасет и проверить наличие целевой переменной."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")
    return df


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удалить идентификаторы и фактические данные, которые дают утечку."""
    # trip_id — технический идентификатор.
    # actual_departure_delay_min и actual_arrival_delay_min становятся известны после поездки.
    # actual_arrival_delay_min напрямую раскрывает delayed, поэтому его нельзя использовать в X.
    existing_columns = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    return df.drop(columns=existing_columns)


def split_features_target(df: pd.DataFrame):
    """Разделить DataFrame на признаки X и целевую переменную y."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is missing")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y
