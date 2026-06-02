from pathlib import Path

import pandas as pd
import pytest

from src.experiments import (
    AblationExperiment,
    BaseExperiment,
    ExperimentData,
    ExperimentResult,
    ForwardSelectionExperiment,
    TrainExperiment,
    get_experiment,
)
from src.experiments import base as base_module


class MinimalExperiment(BaseExperiment):
    name = "minimal"
    test_size = 0.5

    def __init__(self):
        self.received_data = None
        self.steps = []

    def run_experiment(self, data):
        self.steps.append("run_experiment")
        self.received_data = data
        return ExperimentResult(name=self.name, metrics={"accuracy": 1.0})

    def save_outputs(self, result):
        self.steps.append("save_outputs")
        return [Path("artifact.txt")]

    def log_results(self, result):
        self.steps.append("log_results")

    def print_report(self, result):
        self.steps.append("print_report")

    def print_metrics(self, result):
        self.steps.append("print_metrics")


def test_get_experiment_returns_requested_experiment_class():
    assert isinstance(get_experiment("train"), TrainExperiment)
    assert isinstance(get_experiment("ablation"), AblationExperiment)
    assert isinstance(get_experiment("forward-selection"), ForwardSelectionExperiment)


def test_get_experiment_fails_for_unknown_mode():
    with pytest.raises(ValueError, match="Unknown experiment mode"):
        get_experiment("unknown")


def test_base_experiment_prepares_common_data_before_specific_experiment(monkeypatch):
    df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3, 4],
            "actual_departure_delay_min": [0, 1, 0, 2],
            "actual_arrival_delay_min": [0, 5, 0, 6],
            "date": ["2024-01-01"] * 4,
            "time": ["08:00"] * 4,
            "scheduled_departure": ["08:00"] * 4,
            "scheduled_arrival": ["08:30"] * 4,
            "event_type": ["None", "Concert", "None", "Game"],
            "event_attendance_est": [100, 200, 300, 400],
            "weather_condition": ["Clear", "Rain", "Clear", "Rain"],
            "delayed": [0, 1, 0, 1],
        }
    )
    monkeypatch.setattr(base_module, "load_data", lambda: df.copy())
    experiment = MinimalExperiment()

    result = experiment.run()
    data = experiment.received_data

    assert result.name == "minimal"
    assert result.artifact_paths == [Path("artifact.txt")]
    assert result.metrics.keys() >= {"accuracy", "cpu_percent_mean", "cpu_percent_max"}
    assert experiment.steps == [
        "run_experiment",
        "save_outputs",
        "log_results",
        "print_report",
        "print_metrics",
    ]
    assert isinstance(data, ExperimentData)
    for X in [data.X_train, data.X_test]:
        assert "delayed" not in X.columns
        assert "trip_id" not in X.columns
        assert "actual_departure_delay_min" not in X.columns
        assert "actual_arrival_delay_min" not in X.columns
        assert "date" not in X.columns
        assert "time" not in X.columns
        assert "scheduled_departure" not in X.columns
        assert "scheduled_arrival" not in X.columns
    assert (pd.concat([data.X_train, data.X_test]).query("event_type == 'None'")["event_attendance_est"] == 0).all()


def test_experiment_result_adds_only_existing_artifact_entries():
    result = ExperimentResult(name="demo", artifact_paths=[Path("old.txt")])

    result.add_artifacts([Path("new.txt"), None])

    assert result.artifact_paths == [Path("old.txt"), Path("new.txt")]


def test_experiment_data_uses_full_target_for_class_balance_when_available():
    y_train = pd.Series([0, 1])
    full_y = pd.Series([0, 0, 1, 1])
    data = ExperimentData(
        X_train=pd.DataFrame({"x": [1, 2]}),
        y_train=y_train,
        X_test=pd.DataFrame({"x": [3, 4]}),
        y_test=pd.Series([0, 1]),
        y=full_y,
    )

    assert data.class_balance_target().equals(full_y)


def test_experiment_data_falls_back_to_train_target_for_class_balance():
    y_train = pd.Series([0, 1])
    data = ExperimentData(
        X_train=pd.DataFrame({"x": [1, 2]}),
        y_train=y_train,
        X_test=pd.DataFrame({"x": [3, 4]}),
        y_test=pd.Series([0, 1]),
    )

    assert data.class_balance_target().equals(y_train)
