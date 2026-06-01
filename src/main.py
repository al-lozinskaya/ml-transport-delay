import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, ensure_runtime_dirs
from src.evaluate import (
    PUBLIC_METRICS,
    calculate_metrics,
    evaluate_model_comparison,
    save_classification_report,
    save_confusion_matrix,
    save_metrics_bar_chart,
    save_model_comparison_chart,
    save_model_comparison_line_chart,
    save_model_comparison_table,
    save_metrics_json,
)
from src.load_data import drop_leakage_columns, load_data, split_features_target
from src.monitor import SystemMonitor, log_to_mlflow, validate_model_quality
from src.preprocessing import prepare_features
from src.train import _model_name, _save_feature_importance, tune_model


def print_class_balance(y) -> None:
    print("Баланс классов delayed:")
    print(y.value_counts())
    print("\nДоли классов delayed:")
    print(y.value_counts(normalize=True))


def print_model_comparison_report(model_comparison) -> None:
    metric_columns = ["model_name", *PUBLIC_METRICS]
    print("\nИтоговая таблица метрик:")
    print(model_comparison[metric_columns].to_string(index=False))

    for _, row in model_comparison.iterrows():
        print(f"\nМодель: {row['model_name']}")
        print(f"Распределение y_pred: {row['predicted_class_counts']}")
        if row["single_class_prediction"]:
            print(f"Предупреждение: {row['prediction_warning']}")
        print("Confusion matrix:")
        print(row["confusion_matrix"])
        print("Classification report:")
        print(row["classification_report"])


def print_best_model_summary(model_comparison) -> None:
    dummy_mask = model_comparison["model_name"].str.startswith("DummyClassifier")
    dummy_rows = model_comparison[dummy_mask]

    print("\nЛучшие модели по устойчивым метрикам:")
    for metric_name in ["roc_auc", "f1_macro", "balanced_accuracy"]:
        best_row = model_comparison.sort_values(metric_name, ascending=False).iloc[0]
        best_dummy_score = dummy_rows[metric_name].max() if not dummy_rows.empty else float("nan")
        improvement = best_row[metric_name] - best_dummy_score
        print(
            f"{metric_name}: {best_row['model_name']} = {best_row[metric_name]:.4f}; "
            f"лучший Dummy = {best_dummy_score:.4f}; разница = {improvement:.4f}"
        )


def run_pipeline() -> tuple[Pipeline, dict]:
    ensure_runtime_dirs()

    df = load_data()
    df = drop_leakage_columns(df)
    X, y = split_features_target(df)
    X = prepare_features(X)
    print_class_balance(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    with SystemMonitor() as system_monitor:
        search = tune_model(X_train, y_train)
        model_comparison = evaluate_model_comparison(search, X_train, y_train, X_test, y_test)
        best_model = search.best_estimator_
        metrics = calculate_metrics(best_model, X_test, y_test)
        validate_model_quality(metrics)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)

    metrics.update(system_monitor.summary())
    artifact_paths = [
        save_confusion_matrix(best_model, X_test, y_test),
        save_metrics_bar_chart(metrics),
        save_model_comparison_table(model_comparison),
        save_model_comparison_chart(model_comparison),
        save_model_comparison_line_chart(model_comparison),
        save_classification_report(metrics),
        save_metrics_json(metrics),
        _save_feature_importance(best_model),
    ]
    log_to_mlflow(search, metrics, artifact_paths, model_comparison)
    print_model_comparison_report(model_comparison)
    print_best_model_summary(model_comparison)
    
    return best_model, metrics


if __name__ == "__main__":
    _, metrics = run_pipeline()
    print("\nМетрики лучшей модели GridSearchCV:")
    for metric_name in PUBLIC_METRICS:
        print(f"{metric_name}: {metrics[metric_name]:.4f}")
