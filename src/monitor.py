import os
import tempfile
import threading
import warnings
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import mlflow.sklearn
import psutil
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI, MIN_F1, MIN_ROC_AUC
from src.train import _model_name


class SystemMonitor:
    """Контекстный менеджер для мониторинга системных ресурсов (CPU и память) во время выполнения блока кода. Собирает статистику и сохраняет ее для последующего анализа."""

    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.samples: list[dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._stop_event.clear()
        psutil.cpu_percent(interval=None)
        self._thread = threading.Thread(target=self._collect_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 0.5)

    def _collect_loop(self) -> None:
        while not self._stop_event.wait(self.interval):
            self.samples.append(
                {
                    "cpu_percent": float(psutil.cpu_percent(interval=None)),
                    "memory_percent": float(psutil.virtual_memory().percent),
                }
            )

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {
                "cpu_percent_mean": 0.0,
                "cpu_percent_max": 0.0,
                "memory_percent_mean": 0.0,
                "memory_percent_max": 0.0,
            }

        cpu_values = [sample["cpu_percent"] for sample in self.samples]
        memory_values = [sample["memory_percent"] for sample in self.samples]
        return {
            "cpu_percent_mean": sum(cpu_values) / len(cpu_values),
            "cpu_percent_max": max(cpu_values),
            "memory_percent_mean": sum(memory_values) / len(memory_values),
            "memory_percent_max": max(memory_values),
        }


def validate_model_quality(metrics: dict) -> None:
    if metrics["f1"] < MIN_F1:
        warnings.warn(f"F1 below threshold: {metrics['f1']:.3f} < {MIN_F1}", stacklevel=2)
    if metrics["roc_auc"] < MIN_ROC_AUC:
        warnings.warn(
            f"ROC AUC below threshold: {metrics['roc_auc']:.3f} < {MIN_ROC_AUC}",
            stacklevel=2,
        )


def log_to_mlflow(search: GridSearchCV, metrics: dict, artifact_paths: list[Path], model_comparison=None) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    best_estimator: Pipeline = search.best_estimator_
    with mlflow.start_run(run_name=_model_name(best_estimator)):
        mlflow.log_param("model_name", _model_name(best_estimator))
        mlflow.log_params({f"best_{key}": str(value) for key, value in search.best_params_.items()})
        mlflow.log_param(f"cv_best_{search.refit}", search.best_score_)

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        for artifact_path in artifact_paths:
            if artifact_path and artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

        with tempfile.TemporaryDirectory() as tmpdir:
            model_artifact = Path(tmpdir) / "model"
            mlflow.sklearn.save_model(best_estimator, model_artifact)
            mlflow.log_artifacts(str(model_artifact), artifact_path="model")

        if model_comparison is not None:
            log_model_comparison_runs(model_comparison)


def log_model_comparison_runs(comparison) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    for row in comparison.to_dict(orient="records"):
        with mlflow.start_run(run_name=row["model_name"], nested=True):
            mlflow.set_tag("run_type", "model_comparison")
            mlflow.log_param("model_name", row["model_name"])
            mlflow.log_param("best_params", row["best_params"])
            for metric_name in [
                "best_cv_accuracy",
                "best_cv_balanced_accuracy",
                "best_cv_precision",
                "best_cv_recall",
                "best_cv_f1",
                "best_cv_f1_macro",
                "best_cv_roc_auc",
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "f1_macro",
                "roc_auc",
            ]:
                if metric_name in row:
                    mlflow.log_metric(metric_name, float(row[metric_name]))
