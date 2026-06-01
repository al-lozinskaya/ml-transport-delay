import pandas as pd

from src.evaluate import flatten_model_comparison_metrics, save_model_comparison_line_chart


def test_flatten_model_comparison_metrics_uses_model_prefixes():
    comparison = pd.DataFrame(
        [
            {
                "model_name": "LogisticRegression",
                "accuracy": 0.7,
                "f1": 0.8,
                "roc_auc": 0.9,
                "best_cv_f1": 0.75,
            },
            {
                "model_name": "RandomForestClassifier",
                "accuracy": 0.6,
                "f1": 0.65,
                "roc_auc": 0.7,
                "best_cv_f1": 0.68,
            },
        ]
    )

    metrics = flatten_model_comparison_metrics(comparison)

    assert metrics["LogisticRegression_f1"] == 0.8
    assert metrics["LogisticRegression_best_cv_f1"] == 0.75
    assert metrics["RandomForestClassifier_roc_auc"] == 0.7


def test_save_model_comparison_line_chart_creates_file(tmp_path):
    comparison = pd.DataFrame(
        [
            {"model_name": "LogisticRegression", "accuracy": 0.7, "f1": 0.8, "roc_auc": 0.9},
            {"model_name": "RandomForestClassifier", "accuracy": 0.6, "f1": 0.65, "roc_auc": 0.7},
        ]
    )

    output_path = save_model_comparison_line_chart(comparison, tmp_path / "comparison.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
