import json

import pandas as pd

from src import monitor as monitor_module
from src.monitor import (
    MLflowLogger,
    ModelQualityValidator,
    PreprocessingSummaryBuilder,
    SystemMonitor,
)
from src.train import build_training_pipeline


def test_system_monitor_summary_aggregates_samples():
    monitor = SystemMonitor()
    monitor.samples = [
        {"cpu_percent": 10.0, "memory_percent": 40.0},
        {"cpu_percent": 30.0, "memory_percent": 60.0},
    ]

    summary = monitor.summary()

    assert summary == {
        "cpu_percent_mean": 20.0,
        "cpu_percent_max": 30.0,
        "memory_percent_mean": 50.0,
        "memory_percent_max": 60.0,
    }


def test_system_monitor_summary_returns_zeroes_without_samples():
    summary = SystemMonitor().summary()

    assert summary == {
        "cpu_percent_mean": 0.0,
        "cpu_percent_max": 0.0,
        "memory_percent_mean": 0.0,
        "memory_percent_max": 0.0,
    }


def test_mlflow_logger_configures_tracking_uri_and_experiment(monkeypatch):
    calls = {"tracking_uri": [], "experiment": []}

    monkeypatch.setattr(monitor_module.mlflow, "set_tracking_uri", lambda value: calls["tracking_uri"].append(value))
    monkeypatch.setattr(monitor_module.mlflow, "set_experiment", lambda value: calls["experiment"].append(value))

    MLflowLogger(tracking_uri="file:/tmp/mlruns", experiment_name="demo").configure()

    assert calls == {"tracking_uri": ["file:/tmp/mlruns"], "experiment": ["demo"]}


def test_log_numeric_metrics_skips_non_numeric_values(monkeypatch):
    calls = []

    monkeypatch.setattr(monitor_module.mlflow, "log_metric", lambda key, value: calls.append((key, value)))

    MLflowLogger().log_numeric_metrics({"roc_auc": 0.7, "classification_report": "text"})

    assert calls == [("roc_auc", 0.7)]


def test_log_artifacts_logs_existing_paths_only(monkeypatch, tmp_path):
    calls = []
    existing_path = tmp_path / "artifact.csv"
    missing_path = tmp_path / "missing.csv"
    existing_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(monitor_module.mlflow, "log_artifact", lambda path: calls.append(path))

    MLflowLogger().log_artifacts([existing_path, missing_path, None])

    assert calls == [str(existing_path)]


def test_mlflow_logger_logs_common_experiment_run(monkeypatch, tmp_path):
    calls = {"run_name": [], "params": [], "metrics": [], "artifacts": [], "tags": []}

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    artifact_path = tmp_path / "artifact.csv"
    artifact_path.write_text("ok", encoding="utf-8")

    monkeypatch.setattr(monitor_module.mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(monitor_module.mlflow, "set_experiment", lambda value: None)
    monkeypatch.setattr(
        monitor_module.mlflow,
        "start_run",
        lambda run_name: calls["run_name"].append(run_name) or DummyRun(),
    )
    monkeypatch.setattr(monitor_module.mlflow, "log_param", lambda key, value: calls["params"].append((key, value)))
    monkeypatch.setattr(monitor_module.mlflow, "set_tag", lambda key, value: calls["tags"].append((key, value)))
    monkeypatch.setattr(monitor_module.mlflow, "log_metric", lambda key, value: calls["metrics"].append((key, value)))
    monkeypatch.setattr(monitor_module.mlflow, "log_artifact", lambda path: calls["artifacts"].append(path))

    MLflowLogger().log_experiment(
        run_name="forward-selection",
        metrics={"roc_auc": 0.7, "classification_report": "text"},
        artifact_paths=[artifact_path],
        params={"selected_features": "traffic_congestion_index"},
        tags={"run_type": "forward_selection"},
    )

    assert calls["run_name"] == ["forward-selection"]
    assert calls["params"] == [("selected_features", "traffic_congestion_index")]
    assert calls["tags"] == [("run_type", "forward_selection")]
    assert calls["metrics"] == [("roc_auc", 0.7)]
    assert calls["artifacts"] == [str(artifact_path)]


def test_mlflow_logger_logs_model_when_experiment_model_is_passed(monkeypatch):
    calls = []

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return None

    monkeypatch.setattr(monitor_module.mlflow, "set_tracking_uri", lambda value: None)
    monkeypatch.setattr(monitor_module.mlflow, "set_experiment", lambda value: None)
    monkeypatch.setattr(monitor_module.mlflow, "start_run", lambda run_name: DummyRun())
    monkeypatch.setattr(MLflowLogger, "log_preprocessing_and_model", lambda self, model: calls.append(model))

    model = object()
    MLflowLogger().log_experiment(
        run_name="forward-selection",
        metrics={},
        artifact_paths=[],
        model=model,
    )

    assert calls == [model]


def test_build_preprocessing_summary_describes_pipeline_steps():
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, 21.0],
            "transport_type": ["Bus", "Metro"],
        }
    )
    pipeline = build_training_pipeline(X)
    pipeline.fit(X, [0, 1])

    summary = PreprocessingSummaryBuilder().build(pipeline)

    assert summary["numeric_features"] == ["temperature_C"]
    assert summary["categorical_features"] == ["transport_type"]
    assert summary["numeric_pipeline"] == ["SimpleImputer", "StandardScaler"]
    assert summary["categorical_pipeline"] == ["SimpleImputer", "OneHotEncoder"]
    assert summary["selector"]["name"] == "SelectKBest"
    assert summary["model"] == "LogisticRegression"


def test_preprocessing_summary_builder_describes_pipeline_steps():
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, 21.0],
            "transport_type": ["Bus", "Metro"],
        }
    )
    pipeline = build_training_pipeline(X)
    pipeline.fit(X, [0, 1])

    summary = PreprocessingSummaryBuilder().build(pipeline)

    assert summary["numeric_features"] == ["temperature_C"]
    assert summary["categorical_features"] == ["transport_type"]


def test_model_quality_validator_warns_for_low_scores():
    import pytest

    validator = ModelQualityValidator(min_f1=0.6, min_roc_auc=0.6)

    with pytest.warns(UserWarning, match="F1 below threshold"):
        validator.validate({"f1": 0.5, "roc_auc": 0.7})

    with pytest.warns(UserWarning, match="ROC AUC below threshold"):
        validator.validate({"f1": 0.7, "roc_auc": 0.5})


def test_save_preprocessing_summary_creates_json_file(tmp_path):
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, 21.0],
            "transport_type": ["Bus", "Metro"],
        }
    )
    pipeline = build_training_pipeline(X)
    pipeline.fit(X, [0, 1])

    path = PreprocessingSummaryBuilder().save(pipeline, tmp_path / "preprocessing_summary.json")

    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["numeric_features"] == ["temperature_C"]
    assert saved["categorical_features"] == ["transport_type"]


def test_build_preprocessing_summary_handles_disabled_selector():
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, 21.0],
            "transport_type": ["Bus", "Metro"],
        }
    )
    pipeline = build_training_pipeline(X)
    pipeline.set_params(selector="passthrough")
    pipeline.fit(X, [0, 1])

    summary = PreprocessingSummaryBuilder().build(pipeline)

    assert summary["selector"] == {"name": "passthrough"}
