"""Train, compare, and persist the best spam-classification pipeline."""

import json
import logging
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split

from .config import (
    DATA_PATH,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from .data import load_dataset
from .model import build_pipelines
from .preprocessing import clean_text

logger = logging.getLogger(__name__)


def train(
    data_path: Path | str = DATA_PATH,
    model_path: Path | str = MODEL_PATH,
    metrics_path: Path | str = METRICS_PATH,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict:
    """Fit candidate pipelines, select the best by spam F1, and persist it."""
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    logger.info("Loading dataset from %s", data_path)
    df = load_dataset(data_path)
    df["message"] = df["message"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    logger.info("Train=%d Test=%d", len(X_train), len(X_test))

    best = None  # (name, pipeline, accuracy, spam_f1)
    for name, pipe in build_pipelines().items():
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        spam_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        logger.info("%s: accuracy=%.4f spam_f1=%.4f", name, acc, spam_f1)
        print(f"\n===== {name} =====")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        if best is None or spam_f1 > best[3]:
            best = (name, pipe, acc, spam_f1)

    best_name, best_pipe, best_acc, best_f1 = best
    print(f"\nBest model: {best_name} (spam F1={best_f1:.4f})")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(best_pipe, model_path)

    summary = {
        "best_model": best_name,
        "accuracy": round(float(best_acc), 4),
        "spam_f1": round(float(best_f1), 4),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info("Saved model to %s", model_path)
    return summary
