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

MODEL_N_ESTIMATORS = [10, 30, 50, 70, 100, 200]

SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "f1_macro": "f1_macro",
    "roc_auc": "roc_auc",
}

FEATURE_SETS = {
    "all_features": [
        "transport_type",
        "route_id",
        "origin_station",
        "destination_station",
        "weather_condition",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "event_type",
        "event_attendance_est",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "weekday",
        "season",
    ],
    "without_stations": [
        "transport_type",
        "route_id",
        "weather_condition",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "event_type",
        "event_attendance_est",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "weekday",
        "season",
    ],
    "without_route_and_stations": [
        "transport_type",
        "weather_condition",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "event_type",
        "event_attendance_est",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "weekday",
        "season",
    ],
    "weather_time_events_only": [
        "weather_condition",
        "temperature_C",
        "humidity_percent",
        "wind_speed_kmh",
        "precipitation_mm",
        "event_type",
        "event_attendance_est",
        "traffic_congestion_index",
        "holiday",
        "peak_hour",
        "weekday",
        "season",
    ],
}

TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 3

MIN_F1 = 0.6
MIN_ROC_AUC = 0.6

FORWARD_SELECTION_MAX_FEATURES = 8
FORWARD_SELECTION_SCORING = "roc_auc"
FORWARD_SELECTION_MIN_IMPROVEMENT = 0.0

MLFLOW_EXPERIMENT_NAME = "transport-delay-classification"
MLFLOW_DIR = PROJECT_ROOT / "mlruns"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")


def ensure_runtime_dirs() -> None:
    for path in [REPORTS_DIR, FIGURES_DIR, MODELS_DIR, MLFLOW_DIR]:
        path.mkdir(parents=True, exist_ok=True)
