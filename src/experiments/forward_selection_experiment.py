import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, FIGURES_DIR, FORWARD_SELECTION_MAX_FEATURES, FORWARD_SELECTION_MIN_IMPROVEMENT, FORWARD_SELECTION_SCORING, RANDOM_STATE
from src.evaluate import PUBLIC_METRICS, calculate_metrics, evaluate_model_comparison
from src.experiments.base import BaseExperiment, ExperimentData, ExperimentResult
from src.experiments.reporting import print_forward_selection_report
from src.monitor import MLflowLogger
from src.preprocessing import build_preprocessor
from src.train import tune_model


class ForwardSelectionExperiment(BaseExperiment):
    name = "forward-selection"

    def build_forward_selection_pipeline(self, X: pd.DataFrame, model=None) -> Pipeline:
        if model is None:
            model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)

        return Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(X)),
                ("variance", VarianceThreshold()),
                ("model", clone(model)),
            ]
        )

    def run_experiment(self, data: ExperimentData) -> ExperimentResult:
        """Выбрать исходные признаки пошаговым добавлением по метрике качества."""
        X_train = data.X_train
        y_train = data.y_train
        X_test = data.X_test
        y_test = data.y_test
        remaining_features = list(X_train.columns)
        selected_features: list[str] = []
        history_rows = []
        best_score = float("-inf")

        while remaining_features and len(selected_features) < FORWARD_SELECTION_MAX_FEATURES:
            candidate_scores = []
            for feature_name in remaining_features:
                trial_features = [*selected_features, feature_name]
                X_subset = X_train.loc[:, trial_features]
                pipeline = self.build_forward_selection_pipeline(X_subset)
                scores = cross_val_score(
                    pipeline,
                    X_subset,
                    y_train,
                    scoring=FORWARD_SELECTION_SCORING,
                    cv=CV_FOLDS,
                    n_jobs=-1,
                )
                score = float(scores.mean())
                candidate_scores.append((feature_name, score))

            best_feature, next_score = max(candidate_scores, key=lambda item: item[1])
            improvement = next_score - best_score if best_score != float("-inf") else next_score
            if selected_features and improvement < FORWARD_SELECTION_MIN_IMPROVEMENT:
                break

            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            best_score = next_score
            history_rows.append(
                {
                    "step": len(selected_features),
                    "selected_feature": best_feature,
                    "cv_score": next_score,
                    "metric": FORWARD_SELECTION_SCORING,
                    "selected_features": ", ".join(selected_features),
                }
            )

        if not selected_features:
            raise ValueError("Forward Selection did not select any features")

        X_train_selected = X_train.loc[:, selected_features]
        X_test_selected = X_test.loc[:, selected_features]
        search = tune_model(X_train_selected, y_train)
        best_model = search.best_estimator_
        model_comparison = evaluate_model_comparison(
            search,
            X_train_selected,
            y_train,
            X_test_selected,
            y_test,
        )
        metrics = calculate_metrics(best_model, X_test_selected, y_test)

        history = pd.DataFrame(history_rows)

        return ExperimentResult(
            name=self.name,
            metrics=metrics,
            model=best_model,
            tables={
                "forward_selection": history,
                "selected_features": selected_features,
                "search": search,
                "model_comparison": model_comparison,
            },
        )

    def save_outputs(self, result: ExperimentResult):
        path = FIGURES_DIR / "forward_selection_results.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = result.tables["forward_selection"].copy()
        for metric_name in PUBLIC_METRICS:
            rows[f"test_{metric_name}"] = result.metrics[metric_name]
        rows.to_csv(path, index=False)
        return [path]

    def log_results(self, result: ExperimentResult) -> None:
        selected_features = result.tables["selected_features"]
        history = result.tables["forward_selection"]
        MLflowLogger().log_grid_search(
            result.name,
            result.tables["search"],
            result.metrics,
            result.artifact_paths,
            result.tables["model_comparison"],
            params={
                "selected_feature_count": len(selected_features),
                "selected_features": ", ".join(selected_features),
                "forward_selection_scoring": FORWARD_SELECTION_SCORING,
                "forward_selection_max_features": FORWARD_SELECTION_MAX_FEATURES,
                "forward_selection_min_improvement": FORWARD_SELECTION_MIN_IMPROVEMENT,
                "cv_folds": CV_FOLDS,
                "best_cv_score": float(history["cv_score"].iloc[-1]),
            },
            tags={"run_type": "forward_selection"},
        )

    def print_report(self, result: ExperimentResult) -> None:
        print_forward_selection_report(
            result.tables["forward_selection"],
            result.tables["selected_features"],
            result.metrics,
        )
