import json
import os
import tempfile
import threading
import warnings
from dataclasses import dataclass
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
    """Мониторинг CPU/RAM во время выполнения блока кода."""

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


@dataclass
class ModelQualityValidator:
    min_f1: float = MIN_F1
    min_roc_auc: float = MIN_ROC_AUC

    def validate(self, metrics: dict) -> None:
        if metrics["f1"] < self.min_f1:
            warnings.warn(
                f"F1 below threshold: {metrics['f1']:.3f} < {self.min_f1}",
                stacklevel=2,
            )
        if metrics["roc_auc"] < self.min_roc_auc:
            warnings.warn(
                f"ROC AUC below threshold: {metrics['roc_auc']:.3f} < {self.min_roc_auc}",
                stacklevel=2,
            )


class PreprocessingSummaryBuilder:
    def build(self, best_estimator: Pipeline) -> dict:
        preprocessor = best_estimator.named_steps["preprocessor"]
        selector = best_estimator.named_steps.get("selector")
        model = best_estimator.named_steps["model"]

        summary = {
            "preprocessor": preprocessor.__class__.__name__,
            "numeric_features": [],
            "categorical_features": [],
            "numeric_pipeline": [],
            "categorical_pipeline": [],
            "selector": self._selector_summary(selector),
            "model": model.__class__.__name__,
        }

        for name, transformer, columns in preprocessor.transformers_:
            if name == "num":
                summary["numeric_features"] = list(columns)
                summary["numeric_pipeline"] = self._pipeline_step_names(transformer)
            elif name == "cat":
                summary["categorical_features"] = list(columns)
                summary["categorical_pipeline"] = self._pipeline_step_names(transformer)

        return summary

    def save(self, best_estimator: Pipeline, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.build(best_estimator), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return path

    @staticmethod
    def _pipeline_step_names(pipeline) -> list[str]:
        return [step.__class__.__name__ for _, step in pipeline.steps]

    @staticmethod
    def _selector_summary(selector):
        if selector is None:
            return None
        if selector == "passthrough":
            return {"name": "passthrough"}
        return {
            "name": selector.__class__.__name__,
            "k": selector.k,
            "score_func": selector.score_func.__name__,
        }


@dataclass
class MLflowLogger:
    tracking_uri: str = MLFLOW_TRACKING_URI
    experiment_name: str = MLFLOW_EXPERIMENT_NAME
    preprocessing_summary_builder: PreprocessingSummaryBuilder = PreprocessingSummaryBuilder()

    def configure(self) -> None:
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

    def log_numeric_metrics(self, metrics: dict) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

    def log_artifacts(self, artifact_paths: list[Path | None]) -> None:
        for artifact_path in artifact_paths:
            if artifact_path and artifact_path.exists():
                mlflow.log_artifact(str(artifact_path))

    def log_experiment(
        self,
        run_name: str,
        metrics: dict,
        artifact_paths: list[Path | None],
        params: dict | None = None,
        tags: dict | None = None,
        model: Pipeline | None = None,
        model_comparison=None,
    ) -> None:
        self.configure()
        with mlflow.start_run(run_name=run_name):
            self.log_tags(tags or {})
            self.log_params(params or {})
            self.log_numeric_metrics(metrics)
            self.log_artifacts(artifact_paths)
            if model is not None:
                self.log_preprocessing_and_model(model)
            if model_comparison is not None:
                self.log_model_comparison_runs(model_comparison)

    def log_grid_search(
        self,
        run_name,
        search: GridSearchCV,
        metrics: dict,
        artifact_paths: list[Path],
        model_comparison=None,
        params: dict | None = None,
        tags: dict | None = None,
    ) -> None:
        self.configure()

        best_estimator: Pipeline = search.best_estimator_
        model_name = _model_name(best_estimator)
        with mlflow.start_run(run_name=f"{run_name} - {model_name}"):
            self.log_tags(tags or {})
            self.log_params(
                {
                    "model_name": model_name,
                    **{f"best_{key}": str(value) for key, value in search.best_params_.items()},
                    f"cv_best_{search.refit}": search.best_score_,
                    **(params or {}),
                }
            )
            self.log_numeric_metrics(metrics)
            self.log_artifacts(artifact_paths)
            self.log_preprocessing_and_model(best_estimator)

            if model_comparison is not None:
                self.log_model_comparison_runs(model_comparison)

    def log_model_comparison_runs(self, comparison) -> None:
        self.configure()

        for row in comparison.to_dict(orient="records"):
            run_name = (
                f"{row['feature_set']} - {row['model_name']}"
                if "feature_set" in row
                else row["model_name"]
            )
            with mlflow.start_run(run_name=run_name, nested=True):
                self.log_tags({"run_type": "model_comparison"})
                self.log_params(self._model_comparison_params(row))
                self.log_numeric_metrics(self._model_comparison_metrics(row))

    def log_preprocessing_and_model(self, best_estimator: Pipeline) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            preprocessing_summary = self.preprocessing_summary_builder.save(
                best_estimator,
                Path(tmpdir) / "preprocessing_summary.json",
            )
            mlflow.log_artifact(str(preprocessing_summary), artifact_path="preprocessing")

            model_artifact = Path(tmpdir) / "model"
            mlflow.sklearn.save_model(best_estimator, model_artifact)
            mlflow.log_artifacts(str(model_artifact), artifact_path="model")

    @staticmethod
    def log_params(params: dict) -> None:
        for key, value in params.items():
            mlflow.log_param(key, value)

    @staticmethod
    def log_tags(tags: dict) -> None:
        for key, value in tags.items():
            mlflow.set_tag(key, value)

    @staticmethod
    def _model_comparison_metrics(row: dict) -> dict:
        metric_names = [
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
        ]
        return {metric_name: float(row[metric_name]) for metric_name in metric_names if metric_name in row}

    @staticmethod
    def _model_comparison_params(row: dict) -> dict:
        param_names = [
            "feature_set",
            "feature_count",
            "features",
            "model_name",
            "best_params",
        ]
        return {param_name: row[param_name] for param_name in param_names if param_name in row}
