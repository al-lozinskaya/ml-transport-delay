import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _looks_like_date(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    return values.str.match(r"^\d{4}-\d{2}-\d{2}$").mean() >= 0.8


def _looks_like_time(series: pd.Series) -> bool:
    values = series.dropna().astype(str)
    if values.empty:
        return False
    return values.str.match(r"^\d{2}:\d{2}:\d{2}$").mean() >= 0.8


def add_datetime_features(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразовать строковые признаки, которые выглядят как даты или время, 
    в числовые календарные признаки (например, час, день недели, месяц). 
    Это может помочь модели лучше понять временные зависимости в данных."""
    result = df.copy()

    for column in list(result.columns):
        if not pd.api.types.is_object_dtype(result[column]):
            continue

        lower_name = column.lower()
        if any(token in lower_name for token in ["time", "departure", "arrival"]) or _looks_like_time(
            result[column]
        ):
            parsed = pd.to_datetime(result[column].astype(str), format="%H:%M:%S", errors="coerce")
            if parsed.notna().any():
                result[f"{column}_hour"] = parsed.dt.hour
                result = result.drop(columns=[column])
                continue

        if "date" in lower_name or _looks_like_date(result[column]):
            parsed = pd.to_datetime(result[column], format="%Y-%m-%d", errors="coerce")
            if parsed.notna().any():
                result[f"{column}_day_of_week"] = parsed.dt.dayofweek
                result[f"{column}_month"] = parsed.dt.month
                result = result.drop(columns=[column])

    return result


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Построить препроцессор для числовых и категориальных признаков.
    Числовые признаки будут заполнены медианой и стандартизированы,
    а категориальные признаки будут заполнены константой и закодированы с помощью One-Hot Encoding. 
    Это обеспечит, что модель сможет эффективно использовать все типы данных."""
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
    """Преобразовать признаки, добавив календарные признаки и закодировав категориальные переменные."""
    return add_datetime_features(df)
