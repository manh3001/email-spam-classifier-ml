"""Load a trained pipeline and classify messages."""

from pathlib import Path

import joblib

from .config import MODEL_PATH
from .preprocessing import clean_text


class SpamClassifier:
    """Thin wrapper around a fitted TF-IDF + classifier pipeline."""

    def __init__(self, pipeline):
        self.pipeline = pipeline

    @classmethod
    def load(cls, path: Path | str = MODEL_PATH) -> "SpamClassifier":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Model not found at {path}. Run `spam-classify train` first."
            )
        return cls(joblib.load(path))

    def _spam_probability(self, cleaned: str) -> float:
        proba = self.pipeline.predict_proba([cleaned])[0]
        # Class 1 == spam; classes_ gives column order.
        spam_idx = list(self.pipeline.classes_).index(1)
        return float(proba[spam_idx])

    def predict(self, text: str) -> dict:
        cleaned = clean_text(text)
        prob = self._spam_probability(cleaned)
        return {
            "label": "SPAM" if prob >= 0.5 else "HAM",
            "spam_probability": round(prob, 4),
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(t) for t in texts]
