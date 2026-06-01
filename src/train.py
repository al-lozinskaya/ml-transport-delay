import os

import matplotlib.pyplot as plt

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    RANDOM_STATE,
    SCORING,
    MODEL_N_ESTIMATORS
)
from src.preprocessing import build_preprocessor



def build_training_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            # OneHotEncoder возвращает dense-матрицу, чтобы SelectKBest одинаково работал
            # для линейных моделей, деревьев и boosting-классификаторов.
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def _optional_model_grids(k_values: list[int | str]) -> list[dict]:
    optional_grids = []

    try:
        from catboost import CatBoostClassifier

        optional_grids.append(
            {
                "selector__k": k_values,
                "model": [CatBoostClassifier(verbose=0, random_seed=RANDOM_STATE)],
                "model__depth": [4, 6],
                "model__learning_rate": [0.05, 0.1],
            }
        )
    except ImportError:
        pass

    try:
        from xgboost import XGBClassifier

        optional_grids.append(
            {
                "selector__k": k_values,
                "model": [XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)],
                "model__n_estimators": MODEL_N_ESTIMATORS,
                "model__max_depth": [3, 5],
            }
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMClassifier

        optional_grids.append(
            {
                "selector__k": k_values,
                "model": [LGBMClassifier(random_state=RANDOM_STATE, verbose=-1)],
                "model__n_estimators": MODEL_N_ESTIMATORS,
                "model__num_leaves": [15, 31],
            }
        )
    except ImportError:
        pass

    return optional_grids


def build_param_grid(feature_count: int) -> list[dict]:
    k_values = sorted({min(5, feature_count), min(10, feature_count)})
    k_values.append("all")

    param_grid = [
        # DummyClassifier нужен как честная нижняя планка: если реальные модели не лучше
        # такого baseline, значит они почти не нашли полезных закономерностей.
        {
            "selector__k": ["all"],
            "model": [DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)],
        },
        {
            "selector__k": ["all"],
            "model": [DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)],
        },
        {
            "selector__k": k_values,
            "model": [LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 10.0],
        },
        {
            "selector__k": k_values,
            "model": [DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)],
            "model__max_depth": [None, 4, 8, 16],
        },
        {
            "selector__k": k_values,
            "model": [RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)],
            "model__n_estimators": MODEL_N_ESTIMATORS,
            "model__max_depth": [None, 8, 16],
        },
        {
            "selector__k": k_values,
            "model": [ExtraTreesClassifier(class_weight="balanced", random_state=RANDOM_STATE)],
            "model__n_estimators": MODEL_N_ESTIMATORS,
            "model__max_depth": [None, 8, 16],
        },
        {
            "selector__k": k_values,
            "model": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": MODEL_N_ESTIMATORS,
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        {
            "selector__k": k_values,
            "model": [HistGradientBoostingClassifier(random_state=RANDOM_STATE)],
            "model__learning_rate": [0.05, 0.1],
            "model__max_iter": MODEL_N_ESTIMATORS,
        },
    ]

    return param_grid + _optional_model_grids(k_values)


def build_grid_search(pipeline: Pipeline, feature_count: int, refit: str = "roc_auc") -> GridSearchCV:
    return GridSearchCV(
        estimator=pipeline,
        param_grid=build_param_grid(feature_count),
        scoring=SCORING,
        refit=refit,
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
        error_score="raise",
    )


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipeline = build_training_pipeline(X_train)
    search = build_grid_search(pipeline, X_train.shape[1], refit="roc_auc")
    try:
        search.fit(X_train, y_train)
        return search
    except ValueError:
        fallback_search = build_grid_search(pipeline, X_train.shape[1], refit="f1_macro")
        fallback_search.fit(X_train, y_train)
        return fallback_search


def _model_name(best_estimator: Pipeline) -> str:
    return best_estimator.named_steps["model"].__class__.__name__


def _save_feature_importance(best_estimator: Pipeline, path=FIGURES_DIR / "feature_importance.png"):
    model = best_estimator.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    indices = importances.argsort()[-20:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(indices)), importances[indices], color="#3c7d4b")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([f"feature_{index}" for index in indices])
    ax.set_title("Топ 20 важнейших признаков")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

