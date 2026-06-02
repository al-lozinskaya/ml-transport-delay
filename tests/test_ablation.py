import pandas as pd
import pytest

from src.config import FEATURE_SETS
from src.experiments.base import ExperimentResult
from src.experiments.ablation_experiment import AblationExperiment


def test_feature_sets_include_expected_ablation_variants():
    assert set(FEATURE_SETS) == {
        "all_features",
        "without_stations",
        "without_route_and_stations",
        "weather_time_events_only",
    }
    removed_time_features = {"month", "day", "is_weekend", "hour", "minute", "planned_duration_min"}
    for feature_names in FEATURE_SETS.values():
        assert removed_time_features.isdisjoint(feature_names)


def test_select_feature_set_keeps_columns_in_requested_order():
    X = pd.DataFrame(
        {
            "weather_condition": ["Rain"],
            "temperature_C": [12.0],
            "hour": [8],
        }
    )

    selected = AblationExperiment.select_feature_set(X, ["hour", "weather_condition"])

    assert selected.columns.tolist() == ["hour", "weather_condition"]


def test_select_feature_set_fails_for_missing_columns():
    X = pd.DataFrame({"weather_condition": ["Rain"]})

    with pytest.raises(ValueError, match="Missing columns"):
        AblationExperiment.select_feature_set(X, ["weather_condition", "hour"])


def test_ablation_experiment_uses_best_row_as_result_metrics():
    ablation_results = pd.DataFrame(
        [
            {
                "feature_set": "weak",
                "model_name": "LogisticRegression",
                "feature_count": 1,
                "features": '["weather_condition"]',
                "best_model": "LogisticRegression",
                "best_params": "{}",
                "accuracy": 0.5,
                "balanced_accuracy": 0.5,
                "precision": 0.5,
                "recall": 0.5,
                "f1": 0.5,
                "f1_macro": 0.5,
                "roc_auc": 0.4,
            },
            {
                "feature_set": "strong",
                "model_name": "RandomForestClassifier",
                "feature_count": 2,
                "features": '["weather_condition", "traffic_congestion_index"]',
                "best_model": "RandomForestClassifier",
                "best_params": "{}",
                "accuracy": 0.8,
                "balanced_accuracy": 0.75,
                "precision": 0.7,
                "recall": 0.9,
                "f1": 0.8,
                "f1_macro": 0.76,
                "roc_auc": 0.9,
            },
        ]
    )

    class DemoAblationExperiment(AblationExperiment):
        def build_ablation_results(self, data):
            return ablation_results

    result = DemoAblationExperiment().run_experiment(None)

    assert result.metrics == {
        "accuracy": 0.8,
        "balanced_accuracy": 0.75,
        "precision": 0.7,
        "recall": 0.9,
        "f1": 0.8,
        "f1_macro": 0.76,
        "roc_auc": 0.9,
    }
    assert result.tables["best_feature_set"] == "strong"


def test_ablation_logs_best_feature_set_metrics_params_and_artifacts_to_mlflow(monkeypatch, tmp_path):
    calls = []
    artifact_path = tmp_path / "ablation_results.csv"
    artifact_path.write_text("feature_set,roc_auc\nall_features,0.7\n", encoding="utf-8")

    from src.experiments import ablation_experiment as ablation_module

    ablation_results = pd.DataFrame(
        [
            {
                "feature_set": "all_features",
                "model_name": "RandomForestClassifier",
                "feature_count": 3,
                "features": '["a", "b", "c"]',
                "best_model": "RandomForestClassifier",
                "best_params": '{"model": "RandomForestClassifier"}',
                "roc_auc": 0.7,
                "f1_macro": 0.6,
                "balanced_accuracy": 0.65,
            }
        ]
    )
    result = ExperimentResult(
        name="ablation",
        metrics={"roc_auc": 0.7, "f1_macro": 0.6, "balanced_accuracy": 0.65},
        artifact_paths=[artifact_path],
        tables={
            "ablation_results": ablation_results,
            "best_feature_set": "all_features",
        },
    )

    class DummyLogger:
        def log_experiment(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(ablation_module, "MLflowLogger", lambda: DummyLogger())

    AblationExperiment().log_results(result)

    assert calls == [
        {
            "run_name": "ablation - RandomForestClassifier",
            "metrics": {"roc_auc": 0.7, "f1_macro": 0.6, "balanced_accuracy": 0.65},
            "artifact_paths": [artifact_path],
            "params": {
                "feature_set_count": 1,
                "best_feature_set": "all_features",
                "best_feature_count": 3,
                "best_feature_set_features": '["a", "b", "c"]',
                "best_model": "RandomForestClassifier",
                "best_params": '{"model": "RandomForestClassifier"}',
            },
            "tags": {"run_type": "ablation"},
            "model_comparison": ablation_results,
        }
    ]


def test_ablation_builds_model_comparison_rows_for_each_feature_set(monkeypatch):
    calls = []
    from src.experiments import ablation_experiment as ablation_module
    from src.experiments.base import ExperimentData

    X_train = pd.DataFrame({"weather_condition": ["Clear", "Rain"], "temperature_C": [20.0, 10.0]})
    X_test = pd.DataFrame({"weather_condition": ["Clear", "Rain"], "temperature_C": [21.0, 11.0]})
    data = ExperimentData(
        X_train=X_train,
        y_train=pd.Series([0, 1]),
        X_test=X_test,
        y_test=pd.Series([0, 1]),
    )

    class DummySearch:
        best_estimator_ = object()
        best_params_ = {}

    feature_sets = {
        "weather": ["weather_condition"],
        "weather_temp": ["weather_condition", "temperature_C"],
    }

    def fake_tune_model(X_subset, y_subset):
        calls.append(("tune", X_subset.columns.tolist()))
        return DummySearch()

    def fake_evaluate_model_comparison(search, X_train_subset, y_train_subset, X_test_subset, y_test_subset):
        calls.append(("comparison", X_train_subset.columns.tolist()))
        return pd.DataFrame(
            [
                {
                    "model_name": "LogisticRegression",
                    "accuracy": 0.5,
                    "balanced_accuracy": 0.5,
                    "precision": 0.5,
                    "recall": 0.5,
                    "f1": 0.5,
                    "f1_macro": 0.5,
                    "roc_auc": 0.5,
                    "best_params": "{}",
                },
                {
                    "model_name": "RandomForestClassifier",
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.8,
                    "precision": 0.8,
                    "recall": 0.8,
                    "f1": 0.8,
                    "f1_macro": 0.8,
                    "roc_auc": 0.8,
                    "best_params": "{}",
                },
            ]
        )

    monkeypatch.setattr(ablation_module, "FEATURE_SETS", feature_sets)
    monkeypatch.setattr(ablation_module, "tune_model", fake_tune_model)
    monkeypatch.setattr(ablation_module, "evaluate_model_comparison", fake_evaluate_model_comparison)

    results = AblationExperiment().build_ablation_results(data)

    assert calls == [
        ("tune", ["weather_condition"]),
        ("comparison", ["weather_condition"]),
        ("tune", ["weather_condition", "temperature_C"]),
        ("comparison", ["weather_condition", "temperature_C"]),
    ]
    assert results["feature_set"].tolist() == ["weather", "weather", "weather_temp", "weather_temp"]
    assert results["model_name"].tolist() == [
        "LogisticRegression",
        "RandomForestClassifier",
        "LogisticRegression",
        "RandomForestClassifier",
    ]
