import pandas as pd

from src.evaluate import calculate_prediction_diagnostics, flatten_model_comparison_metrics, save_model_comparison_line_chart


class ConstantModel:
    def predict(self, X):
        return [1 for _ in range(len(X))]

    def predict_proba(self, X):
        return [[0.2, 0.8] for _ in range(len(X))]


def test_calculate_prediction_diagnostics_warns_for_single_predicted_class():
    X = pd.DataFrame({"feature": [0, 1, 2, 3]})
    y = pd.Series([0, 1, 0, 1])

    diagnostics = calculate_prediction_diagnostics(ConstantModel(), X, y)

    assert diagnostics["predicted_class_counts"] == {1: 4}
    assert diagnostics["single_class_prediction"] is True
    assert diagnostics["warning"]


def test_flatten_model_comparison_metrics_uses_model_prefixes():
    comparison = pd.DataFrame(
        [
            {
                "model_name": "LogisticRegression",
                "accuracy": 0.7,
                "balanced_accuracy": 0.6,
                "f1_macro": 0.65,
                "roc_auc": 0.9,
                "best_cv_roc_auc": 0.75,
            },
            {
                "model_name": "RandomForestClassifier",
                "accuracy": 0.6,
                "balanced_accuracy": 0.55,
                "f1_macro": 0.58,
                "roc_auc": 0.7,
                "best_cv_roc_auc": 0.68,
            },
        ]
    )

    metrics = flatten_model_comparison_metrics(comparison)

    assert metrics["LogisticRegression_f1_macro"] == 0.65
    assert metrics["LogisticRegression_best_cv_roc_auc"] == 0.75
    assert metrics["RandomForestClassifier_roc_auc"] == 0.7


def test_save_model_comparison_line_chart_creates_file(tmp_path):
    comparison = pd.DataFrame(
        [
            {
                "model_name": "LogisticRegression",
                "accuracy": 0.7,
                "balanced_accuracy": 0.6,
                "f1_macro": 0.65,
                "roc_auc": 0.9,
            },
            {
                "model_name": "RandomForestClassifier",
                "accuracy": 0.6,
                "balanced_accuracy": 0.55,
                "f1_macro": 0.58,
                "roc_auc": 0.7,
            },
        ]
    )

    output_path = save_model_comparison_line_chart(comparison, tmp_path / "comparison.png")

    assert output_path.exists()
    assert output_path.stat().st_size > 0
