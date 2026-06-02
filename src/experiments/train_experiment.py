import joblib

from src.config import MODEL_PATH
from src.evaluate import (
    calculate_metrics,
    evaluate_model_comparison,
    save_classification_report,
    save_confusion_matrix,
    save_metrics_bar_chart,
    save_metrics_json,
    save_model_comparison_chart,
    save_model_comparison_line_chart,
    save_model_comparison_table,
)
from src.experiments.base import BaseExperiment, ExperimentData, ExperimentResult
from src.experiments.reporting import print_best_model_summary, print_model_comparison_report
from src.monitor import MLflowLogger, ModelQualityValidator
from src.train import _save_feature_importance, tune_model


class TrainExperiment(BaseExperiment):
    name = "train"

    def run_experiment(self, data: ExperimentData) -> ExperimentResult:
        search = tune_model(data.X_train, data.y_train)
        model_comparison = evaluate_model_comparison(
            search,
            data.X_train,
            data.y_train,
            data.X_test,
            data.y_test,
        )
        best_model = search.best_estimator_
        metrics = calculate_metrics(best_model, data.X_test, data.y_test)
        ModelQualityValidator().validate(metrics)

        return ExperimentResult(
            name=self.name,
            metrics=metrics,
            model=best_model,
            tables={
                "model_comparison": model_comparison,
                "search": search,
                "X_test": data.X_test,
                "y_test": data.y_test,
            },
        )

    def save_outputs(self, result: ExperimentResult):
        best_model = result.model
        X_test = result.tables["X_test"]
        y_test = result.tables["y_test"]
        model_comparison = result.tables["model_comparison"]

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

        return [
            save_confusion_matrix(best_model, X_test, y_test),
            save_metrics_bar_chart(result.metrics),
            save_model_comparison_table(model_comparison),
            save_model_comparison_chart(model_comparison),
            save_model_comparison_line_chart(model_comparison),
            save_classification_report(result.metrics),
            save_metrics_json(result.metrics),
            _save_feature_importance(best_model),
        ]

    def log_results(self, result: ExperimentResult) -> None:
        MLflowLogger().log_grid_search(
            result.name,
            result.tables["search"],
            result.metrics,
            result.artifact_paths,
            result.tables["model_comparison"],
        )

    def print_report(self, result: ExperimentResult) -> None:
        model_comparison = result.tables["model_comparison"]
        print_model_comparison_report(model_comparison)
        print_best_model_summary(model_comparison)
