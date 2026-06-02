from src.evaluate import PUBLIC_METRICS


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


def print_ablation_report(ablation_results) -> None:
    metric_columns = ["feature_set", "best_model", *PUBLIC_METRICS]
    print("\nAblation test по наборам признаков:")
    print(ablation_results[metric_columns].to_string(index=False))


def print_forward_selection_report(history, selected_features, metrics) -> None:
    print("\nForward Selection:")
    print(history.to_string(index=False))
    print("\nВыбранные признаки:")
    for feature_name in selected_features:
        print(f"- {feature_name}")
    print("\nМетрики Forward Selection на test:")
    for metric_name in PUBLIC_METRICS:
        print(f"{metric_name}: {metrics[metric_name]:.4f}")
