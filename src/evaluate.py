import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import FIGURES_DIR


def calculate_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }


def _public_metrics(metrics: dict) -> dict:
    return {
        key: metrics[key]
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]
    }


def _clean_params(params: dict) -> dict:
    cleaned = {}
    for key, value in params.items():
        if key == "model":
            cleaned[key] = value.__class__.__name__
        else:
            cleaned[key] = str(value)
    return cleaned


def evaluate_model_comparison(search, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """Evaluate the best searched configuration for each model family on the holdout set."""
    best_by_model: dict[str, dict] = {}
    for params, mean_score in zip(search.cv_results_["params"], search.cv_results_["mean_test_score"]):
        model_name = params["model"].__class__.__name__
        current = best_by_model.get(model_name)
        if current is None or mean_score > current["best_cv_f1"]:
            best_by_model[model_name] = {
                "params": params,
                "best_cv_f1": float(mean_score),
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
                "best_cv_f1": candidate["best_cv_f1"],
                **_public_metrics(metrics),
                "best_params": json.dumps(_clean_params(candidate["params"]), ensure_ascii=False),
            }
        )

    return pd.DataFrame(rows).sort_values(["f1", "roc_auc"], ascending=False).reset_index(drop=True)


def flatten_model_comparison_metrics(comparison: pd.DataFrame) -> dict[str, float]:
    metric_names = [
        metric_name
        for metric_name in ["best_cv_f1", "accuracy", "precision", "recall", "f1", "roc_auc"]
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
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
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
    ax.bar([value - width / 2 for value in x], comparison["f1"], width=width, label="f1", color="#2f6f9f")
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
        for metric_name in ["accuracy", "precision", "recall", "f1", "roc_auc"]
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
