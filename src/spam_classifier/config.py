"""Central configuration: paths and default hyperparameters."""

from pathlib import Path

# Project root = two levels up from this file (src/spam_classifier/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "spam_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

TFIDF_KWARGS = {"stop_words": "english"}
