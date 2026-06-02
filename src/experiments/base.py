from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TEST_SIZE, ensure_runtime_dirs
from src.evaluate import PUBLIC_METRICS
from src.load_data import drop_leakage_columns, load_data, split_features_target
from src.monitor import SystemMonitor
from src.preprocessing import prepare_features

ArtifactPath = Path | None


@dataclass
class ExperimentData:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    y: pd.Series | None = None

    def class_balance_target(self) -> pd.Series:
        return self.y if self.y is not None else self.y_train


@dataclass
class ExperimentResult:
    name: str
    metrics: dict[str, Any] = field(default_factory=dict)
    artifact_paths: list[ArtifactPath] = field(default_factory=list)
    model: Pipeline | None = None
    tables: dict[str, Any] = field(default_factory=dict)

    def add_artifacts(self, artifact_paths: list[ArtifactPath]) -> None:
        self.artifact_paths.extend(path for path in artifact_paths if path is not None)


class BaseExperiment(ABC):
    name: str
    test_size = TEST_SIZE
    random_state = RANDOM_STATE

    def run(self) -> ExperimentResult:
        ensure_runtime_dirs()
        data = self.prepare_experiment_data()
        self.print_class_balance(data.class_balance_target())
        result = self.execute_experiment(data)
        result.artifact_paths.extend(path for path in self.save_outputs(result) if path is not None)
        self.log_results(result)
        self.print_report(result)
        self.print_metrics(result)
        return result

    def split_train_test(self, X: pd.DataFrame, y: pd.Series) -> ExperimentData:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        return ExperimentData(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, y=y)

    def prepare_experiment_data(self) -> ExperimentData:
        df = load_data()
        df = drop_leakage_columns(df)
        X, y = split_features_target(df)
        X = prepare_features(X)
        return self.split_train_test(X, y)

    def execute_experiment(self, data: ExperimentData) -> ExperimentResult:
        with SystemMonitor() as system_monitor:
            result = self.run_experiment(data)
        result.metrics.update(system_monitor.summary())
        return result

    def print_class_balance(self, y: pd.Series) -> None:
        print("Баланс классов delayed:")
        print(y.value_counts())
        print("\nДоли классов delayed:")
        print(y.value_counts(normalize=True))

    def save_outputs(self, result: ExperimentResult) -> list[ArtifactPath]:
        return []

    def log_results(self, result: ExperimentResult) -> None:
        pass

    def print_report(self, result: ExperimentResult) -> None:
        pass

    def print_metrics(self, result: ExperimentResult) -> None:
        if not result.metrics:
            print(f"\nМетрики эксперимента {result.name} не рассчитаны.")
            return

        print(f"\nМетрики эксперимента {result.name}:")
        for metric_name in PUBLIC_METRICS:
            if metric_name in result.metrics:
                print(f"{metric_name}: {result.metrics[metric_name]:.4f}")

    @abstractmethod
    def run_experiment(self, data: ExperimentData) -> ExperimentResult:
        pass
