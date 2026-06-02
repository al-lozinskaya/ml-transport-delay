import pandas as pd
import numpy as np

from src.experiments.base import ExperimentData
from src.experiments.forward_selection_experiment import ForwardSelectionExperiment


def test_forward_selection_experiment_returns_original_feature_names(monkeypatch):
    X_train = pd.DataFrame(
        {
            "traffic_congestion_index": [10, 20, 80, 90, 15, 25, 85, 95],
            "weather_condition": ["Clear", "Clear", "Rain", "Rain", "Clear", "Clear", "Rain", "Rain"],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    X_test = pd.DataFrame(
        {
            "traffic_congestion_index": [12, 88, 18, 92],
            "weather_condition": ["Clear", "Rain", "Clear", "Rain"],
        }
    )
    y_test = pd.Series([0, 1, 0, 1])
    from src.experiments import forward_selection_experiment as forward_module

    class DummyModel:
        def predict(self, X):
            return y_test.to_numpy()

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    class DummySearch:
        best_estimator_ = DummyModel()

    monkeypatch.setattr(forward_module, "tune_model", lambda X_subset, y_subset: DummySearch())
    monkeypatch.setattr(
        forward_module,
        "evaluate_model_comparison",
        lambda search, X_train_subset, y_train_subset, X_test_subset, y_test_subset: pd.DataFrame(
            {"model_name": ["LogisticRegression"], "roc_auc": [1.0]}
        ),
    )

    experiment = ForwardSelectionExperiment()
    experiment.test_size = 0.5
    result = experiment.run_experiment(
        ExperimentData(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
        )
    )
    selected_features = result.tables["selected_features"]
    history = result.tables["forward_selection"]

    assert selected_features
    assert set(selected_features).issubset({"traffic_congestion_index", "weather_condition"})
    assert history["selected_feature"].tolist() == selected_features
    assert "roc_auc" in result.metrics


def test_forward_selection_logs_metrics_params_and_artifacts_to_mlflow(monkeypatch, tmp_path):
    calls = []

    artifact_path = tmp_path / "forward_selection_results.csv"
    artifact_path.write_text("step,selected_feature\n1,traffic_congestion_index\n", encoding="utf-8")
    from src.experiments import forward_selection_experiment as forward_module

    result = forward_module.ExperimentResult(
        name="forward-selection",
        metrics={"roc_auc": 0.7, "classification_report": "text"},
        model=object(),
        artifact_paths=[artifact_path],
        tables={
            "selected_features": ["traffic_congestion_index"],
            "forward_selection": pd.DataFrame({"cv_score": [0.65]}),
        },
    )

    class DummyLogger:
        def log_grid_search(self, *args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})

    class DummySearch:
        pass

    model_comparison = pd.DataFrame({"model_name": ["LogisticRegression"]})
    search = DummySearch()

    result.tables["search"] = search
    result.tables["model_comparison"] = model_comparison

    monkeypatch.setattr(forward_module, "MLflowLogger", lambda: DummyLogger())

    ForwardSelectionExperiment().log_results(result)

    assert calls == [
        {
            "args": (
                "forward-selection",
                search,
                {"roc_auc": 0.7, "classification_report": "text"},
                [artifact_path],
                model_comparison,
            ),
            "kwargs": {
                "params": {
                    "selected_feature_count": 1,
                    "selected_features": "traffic_congestion_index",
                    "forward_selection_scoring": forward_module.FORWARD_SELECTION_SCORING,
                    "forward_selection_max_features": forward_module.FORWARD_SELECTION_MAX_FEATURES,
                    "forward_selection_min_improvement": forward_module.FORWARD_SELECTION_MIN_IMPROVEMENT,
                    "cv_folds": forward_module.CV_FOLDS,
                    "best_cv_score": 0.65,
                },
                "tags": {"run_type": "forward_selection"},
            },
        }
    ]


def test_forward_selection_runs_common_model_pool_after_selecting_features(monkeypatch):
    calls = {"tune": [], "comparison": []}

    X_train = pd.DataFrame(
        {
            "traffic_congestion_index": [10, 20, 80, 90, 15, 25, 85, 95],
            "weather_condition": ["Clear", "Clear", "Rain", "Rain", "Clear", "Clear", "Rain", "Rain"],
        }
    )
    y_train = pd.Series([0, 0, 1, 1, 0, 0, 1, 1])
    X_test = pd.DataFrame(
        {
            "traffic_congestion_index": [12, 88, 18, 92],
            "weather_condition": ["Clear", "Rain", "Clear", "Rain"],
        }
    )
    y_test = pd.Series([0, 1, 0, 1])

    from src.experiments import forward_selection_experiment as forward_module

    class DummyModel:
        def predict(self, X):
            return y_test.to_numpy()

        def predict_proba(self, X):
            return np.array([[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]])

    class DummySearch:
        best_estimator_ = DummyModel()

    def fake_tune_model(X_subset, y_subset):
        calls["tune"].append(X_subset.columns.tolist())
        return DummySearch()

    def fake_evaluate_model_comparison(search, X_train_subset, y_train_subset, X_test_subset, y_test_subset):
        calls["comparison"].append(X_train_subset.columns.tolist())
        return pd.DataFrame({"model_name": ["LogisticRegression"], "roc_auc": [1.0]})

    monkeypatch.setattr(forward_module, "tune_model", fake_tune_model)
    monkeypatch.setattr(forward_module, "evaluate_model_comparison", fake_evaluate_model_comparison)

    result = ForwardSelectionExperiment().run_experiment(
        ExperimentData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    )

    assert calls["tune"] == [result.tables["selected_features"]]
    assert calls["comparison"] == [result.tables["selected_features"]]
    assert result.tables["search"].best_estimator_ is result.model
    assert "model_comparison" in result.tables
