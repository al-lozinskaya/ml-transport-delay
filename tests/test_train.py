from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.train import build_grid_search, build_param_grid, build_training_pipeline, get_model_feature_names


def test_param_grid_includes_baselines_and_imbalance_aware_models():
    param_grid = build_param_grid(feature_count=12)
    models = [grid["model"][0] for grid in param_grid]

    model_types = [model.__class__ for model in models]
    assert DummyClassifier in model_types
    assert LogisticRegression in model_types
    assert DecisionTreeClassifier in model_types
    assert RandomForestClassifier in model_types
    assert ExtraTreesClassifier in model_types
    assert GradientBoostingClassifier in model_types
    assert HistGradientBoostingClassifier in model_types

    logistic = next(model for model in models if isinstance(model, LogisticRegression))
    decision_tree = next(model for model in models if isinstance(model, DecisionTreeClassifier))
    random_forest = next(model for model in models if isinstance(model, RandomForestClassifier))
    extra_trees = next(model for model in models if isinstance(model, ExtraTreesClassifier))

    assert logistic.class_weight == "balanced"
    assert decision_tree.class_weight == "balanced"
    assert random_forest.class_weight == "balanced"
    assert extra_trees.class_weight == "balanced"


def test_param_grid_includes_both_dummy_strategies():
    param_grid = build_param_grid(feature_count=12)
    dummy_strategies = {
        grid["model"][0].strategy
        for grid in param_grid
        if isinstance(grid["model"][0], DummyClassifier)
    }

    assert dummy_strategies == {"most_frequent", "stratified"}


def test_param_grid_compares_models_with_and_without_select_k_best():
    param_grid = build_param_grid(feature_count=12)
    logistic_grids = [
        grid
        for grid in param_grid
        if isinstance(grid["model"][0], LogisticRegression)
    ]

    selector_values = [grid["selector"][0] for grid in logistic_grids]

    assert "passthrough" in selector_values
    assert any(isinstance(selector, SelectKBest) for selector in selector_values)
    assert all(
        ("selector__k" in grid) == isinstance(grid["selector"][0], SelectKBest)
        for grid in logistic_grids
    )


def test_grid_search_uses_multiple_metrics_and_refits_by_roc_auc():
    pipeline = build_training_pipeline(__import__("pandas").DataFrame({"feature": [0, 1]}))
    search = build_grid_search(pipeline, feature_count=2)

    assert search.scoring == {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f1_macro": "f1_macro",
        "roc_auc": "roc_auc",
    }
    assert search.refit == "roc_auc"


def test_get_model_feature_names_returns_real_names_after_preprocessing_and_selection():
    pd = __import__("pandas")
    X = pd.DataFrame(
        {
            "temperature_C": [20.0, 21.0, 22.0, 23.0],
            "traffic_congestion_index": [30, 45, 60, 80],
            "transport_type": ["Bus", "Metro", "Bus", "Tram"],
        }
    )
    pipeline = build_training_pipeline(X)
    pipeline.set_params(selector__k=2)
    pipeline.fit(X, [0, 1, 0, 1])

    feature_names = get_model_feature_names(pipeline)

    assert len(feature_names) == 2
    assert all(not name.startswith("feature_") for name in feature_names)
    assert any(
        name in {"temperature_C", "traffic_congestion_index"} or name.startswith("transport_type_")
        for name in feature_names
    )
