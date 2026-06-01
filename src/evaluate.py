import json
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR


PUBLIC_METRICS = ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "f1_macro", "roc_auc"]
CV_METRICS = [f"best_cv_{metric_name}" for metric_name in PUBLIC_METRICS]


def _positive_class_proba(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    return None


def calculate_prediction_diagnostics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    prediction_counts = pd.Series(y_pred).value_counts().sort_index().to_dict()
    single_class_prediction = len(prediction_counts) == 1
    warning = None
    if single_class_prediction:
        warning = "Модель предсказывает только один класс."
        warnings.warn(warning, stacklevel=2)

    return {
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
        "predicted_class_counts": prediction_counts,
        "single_class_prediction": single_class_prediction,
        "warning": warning,
    }


def calculate_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = _positive_class_proba(model, X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        # Accuracy и обычный f1 могут выглядеть хорошо при дисбалансе классов,
        # если модель почти всегда выбирает самый частый класс.
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        # f1_macro, balanced_accuracy и roc_auc важнее для сравнения на
        # несбалансированных данных: они сильнее штрафуют игнорирование редкого класса.
        "f1_macro": f1_score(y_test, y_pred, average="macro", zero_division=0),
    }
    metrics["roc_auc"] = roc_auc_score(y_test, y_proba) if y_proba is not None else float("nan")
    metrics.update(calculate_prediction_diagnostics(model, X_test, y_test))
    return metrics


def _public_metrics(metrics: dict) -> dict:
    return {key: metrics[key] for key in PUBLIC_METRICS}


def _clean_params(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        if key == "model":
            cleaned[key] = value.__class__.__name__
        else:
            cleaned[key] = str(value)
    return cleaned


def _candidate_name(model) -> str:
    model_name = model.__class__.__name__
    if hasattr(model, "strategy"):
        return f"{model_name}_{model.strategy}"
    if getattr(model, "class_weight", None):
        return f"{model_name}_{model.class_weight}"
    return model_name


def _candidate_cv_scores(search, index: int) -> dict:
    scores = {}
    for metric_name in PUBLIC_METRICS:
        key = f"mean_test_{metric_name}"
        if key in search.cv_results_:
            scores[f"best_cv_{metric_name}"] = float(search.cv_results_[key][index])
    return scores


def evaluate_model_comparison(search, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Оценить лучшую конфигурацию каждого семейства моделей на holdout-выборке."""
    best_by_model: dict[str, dict] = {}
    score_key = f"mean_test_{search.refit}" if isinstance(search.refit, str) else "mean_test_roc_auc"

    for index, params in enumerate(search.cv_results_["params"]):
        model_name = _candidate_name(params["model"])
        mean_score = search.cv_results_[score_key][index]
        current = best_by_model.get(model_name)
        if current is None or mean_score > current["refit_score"]:
            best_by_model[model_name] = {
                "params": params,
                "refit_score": float(mean_score),
                "cv_scores": _candidate_cv_scores(search, index),
            }

    rows = []
    for model_name, candidate in best_by_model.items():
        model = clone(search.estimator)
        model.set_params(**candidate["params"])
        model.fit(X_train, y_train)
        metrics = calculate_metrics(model, X_test, y_test)

        rows.append(
            {
                "model_name": model_name,
                **candidate["cv_scores"],
                **_public_metrics(metrics),
                "confusion_matrix": metrics["confusion_matrix"],
                "classification_report": metrics["classification_report"],
                "predicted_class_counts": metrics["predicted_class_counts"],
                "single_class_prediction": metrics["single_class_prediction"],
                "prediction_warning": metrics["warning"],
                "best_params": json.dumps(_clean_params(candidate["params"]), ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows).sort_values(["roc_auc", "f1_macro", "balanced_accuracy"], ascending=False).reset_index(drop=True)


def flatten_model_comparison_metrics(comparison: pd.DataFrame) -> dict[str, float]:
    metric_names = [
        metric_name
        for metric_name in [*CV_METRICS, *PUBLIC_METRICS]
        if metric_name in comparison.columns
    ]
    metrics = {}
    for row in comparison.to_dict(orient="records"):
        model_name = row["model_name"]
        for metric_name in metric_names:
            metrics[f"{model_name}_{metric_name}"] = float(row[metric_name])
    return metrics


def save_confusion_matrix(model, X_test, y_test, path=FIGURES_DIR / "confusion_matrix.png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    display = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
    display.figure_.tight_layout()
    display.figure_.savefig(path, dpi=150)
    plt.close(display.figure_)
    return path


def save_metrics_bar_chart(metrics: dict, path=FIGURES_DIR / "metrics.png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = PUBLIC_METRICS
    values = [metrics[name] for name in metric_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(metric_names, values, color="#2f6f9f")
    ax.set_ylim(0, 1)
    ax.set_title("Best model metrics")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_model_comparison_table(comparison: pd.DataFrame, path=FIGURES_DIR / "model_comparison.csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(path, index=False)
    return path


def save_model_comparison_chart(comparison: pd.DataFrame, path=FIGURES_DIR / "model_comparison.png"):
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(comparison))
    width = 0.35
    ax.bar([value - width / 2 for value in x], comparison["f1_macro"], width=width, label="f1_macro", color="#2f6f9f")
    ax.bar(
        [value + width / 2 for value in x],
        comparison["roc_auc"],
        width=width,
        label="roc_auc",
        color="#3c7d4b",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["model_name"], rotation=15, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_model_comparison_line_chart(comparison: pd.DataFrame, path=FIGURES_DIR / "model_comparison_lines.png"):
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [
        metric_name
        for metric_name in PUBLIC_METRICS
        if metric_name in comparison.columns
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    for _, row in comparison.iterrows():
        values = [row[metric_name] for metric_name in metric_names]
        ax.plot(metric_names, values, marker="o", linewidth=2, label=row["model_name"])

    ax.set_ylim(0, 1)
    ax.set_title("Model metrics comparison")
    ax.set_ylabel("score")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def save_classification_report(metrics: dict, path=FIGURES_DIR / "classification_report.txt"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(metrics["classification_report"], encoding="utf-8")
    return path


def save_metrics_json(metrics: dict, path=FIGURES_DIR / "metrics.json"):
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {key: value for key, value in metrics.items() if key != "classification_report"}
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    return path
