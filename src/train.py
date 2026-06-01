import os

import matplotlib.pyplot as plt

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    RANDOM_STATE,
)
from src.preprocessing import build_preprocessor


def build_training_pipeline(X: pd.DataFrame) -> Pipeline:
    preprocessor = build_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            # OneHotEncoder is configured with dense output so SelectKBest can score all features
            # consistently across linear, forest, and boosting models.
            ("variance", VarianceThreshold()),
            ("selector", SelectKBest(score_func=f_classif)),
            ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )


def build_param_grid(feature_count: int) -> list[dict]:
    k_values = sorted({min(5, feature_count), min(10, feature_count)})
    k_values.append("all")

    return [
        {
            "selector__k": k_values,
            "model": [LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)],
            "model__C": [0.1, 1.0, 10.0],
        },
        {
            "selector__k": k_values,
            "model": [RandomForestClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": [10, 30, 50, 70,100, 200],
            "model__max_depth": [None, 8, 16],
        },
        {
            "selector__k": k_values,
            "model": [GradientBoostingClassifier(random_state=RANDOM_STATE)],
            "model__n_estimators": [10, 30, 50, 70,100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    ]


def tune_model(X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
    pipeline = build_training_pipeline(X_train)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=build_param_grid(X_train.shape[1]),
        scoring="f1",
        cv=CV_FOLDS,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    return search


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

