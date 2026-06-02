import json

import pandas as pd

from src.config import FEATURE_SETS, FIGURES_DIR
from src.evaluate import PUBLIC_METRICS, evaluate_model_comparison
from src.experiments.base import BaseExperiment, ExperimentData, ExperimentResult
from src.experiments.reporting import print_ablation_report
from src.monitor import MLflowLogger
from src.train import tune_model


class AblationExperiment(BaseExperiment):
    name = "ablation"

    @staticmethod
    def select_feature_set(X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
        missing_columns = [column for column in feature_names if column not in X.columns]
        if missing_columns:
            raise ValueError(f"Missing columns for feature set: {missing_columns}")
        return X.loc[:, feature_names]

    def build_ablation_results(self, data: ExperimentData) -> pd.DataFrame:
        rows = []

        for feature_set_name, feature_names in FEATURE_SETS.items():
            X_train_subset = self.select_feature_set(data.X_train, feature_names)
            X_test_subset = self.select_feature_set(data.X_test, feature_names)

            search = tune_model(X_train_subset, data.y_train)
            model_comparison = evaluate_model_comparison(
                search,
                X_train_subset,
                data.y_train,
                X_test_subset,
                data.y_test,
            )
            model_comparison.insert(0, "feature_set", feature_set_name)
            model_comparison.insert(1, "feature_count", len(feature_names))
            model_comparison.insert(2, "features", json.dumps(feature_names, ensure_ascii=False))
            model_comparison["best_model"] = model_comparison["model_name"]
            rows.extend(model_comparison.to_dict(orient="records"))

        return pd.DataFrame(rows)

    def run_experiment(self, data: ExperimentData) -> ExperimentResult:
        ablation_results = self.build_ablation_results(data).sort_values(
            ["roc_auc", "f1_macro", "balanced_accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        best_row = ablation_results.iloc[0]
        metrics = {metric_name: float(best_row[metric_name]) for metric_name in PUBLIC_METRICS}

        return ExperimentResult(
            name=self.name,
            metrics=metrics,
            tables={
                "ablation_results": ablation_results,
                "best_feature_set": best_row["feature_set"],
            },
        )

    def save_outputs(self, result: ExperimentResult):
        path = FIGURES_DIR / "ablation_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        result.tables["ablation_results"].to_csv(path, index=False)
        return [path]

    def log_results(self, result: ExperimentResult) -> None:
        ablation_results = result.tables["ablation_results"]
        best_row = ablation_results.iloc[0]
        MLflowLogger().log_experiment(
            run_name=f"{result.name} - {best_row['best_model']}",
            metrics=result.metrics,
            artifact_paths=result.artifact_paths,
            params={
                "feature_set_count": ablation_results["feature_set"].nunique(),
                "best_feature_set": best_row["feature_set"],
                "best_feature_count": int(best_row["feature_count"]),
                "best_feature_set_features": best_row["features"],
                "best_model": best_row["best_model"],
                "best_params": best_row["best_params"],
            },
            tags={"run_type": "ablation"},
            model_comparison=ablation_results,
        )

    def print_report(self, result: ExperimentResult) -> None:
        print_ablation_report(result.tables["ablation_results"])
