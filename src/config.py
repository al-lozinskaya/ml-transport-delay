import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

DATA_PATH = DATA_DIR / "public_transport_delays.csv"
MODEL_PATH = MODELS_DIR / "best_model.joblib"

TARGET_COLUMN = "delayed"
LEAKAGE_COLUMNS = [
    "trip_id",
    "actual_departure_delay_min",
    "actual_arrival_delay_min",
]

MODEL_N_ESTIMATORS = [10, 30, 50, 70,100, 200]

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

MIN_F1 = 0.6
MIN_ROC_AUC = 0.6

MLFLOW_EXPERIMENT_NAME = "transport-delay-classification"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def ensure_runtime_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR, MLFLOW_DIR]:
        path.mkdir(parents=True, exist_ok=True)
