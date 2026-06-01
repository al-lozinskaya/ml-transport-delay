from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.train import build_grid_search, build_param_grid, build_training_pipeline


def test_param_grid_uses_balanced_logistic_and_random_forest_models():
    param_grid = build_param_grid(feature_count=12)

    models = [grid["model"][0] for grid in param_grid]
    model_labels = [
        (model.__class__.__name__, getattr(model, "class_weight", None))
        for model in models
    ]

    assert model_labels == [
        ("LogisticRegression", "balanced"),
        ("RandomForestClassifier", "balanced"),
        ("RandomForestClassifier", "balanced_subsample"),
    ]
    assert all(not isinstance(model, LogisticRegression) or model.class_weight == "balanced" for model in models)
    assert all(
        not isinstance(model, RandomForestClassifier) or model.class_weight in {"balanced", "balanced_subsample"}
        for model in models
    )


def test_tune_model_compares_by_balanced_accuracy():
    pipeline = build_training_pipeline(__import__("pandas").DataFrame({"feature": [0, 1]}))
    search = build_grid_search(pipeline, feature_count=2)

    assert search.scoring == {
        "balanced_accuracy": "balanced_accuracy",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
    }
    assert search.refit == "balanced_accuracy"
