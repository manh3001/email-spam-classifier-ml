"""Factory for the candidate classification pipelines."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, TFIDF_KWARGS


def build_pipelines() -> dict[str, Pipeline]:
    """Return named, unfitted candidate pipelines (TF-IDF + classifier)."""
    return {
        "naive_bayes": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                ("clf", MultinomialNB()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("tfidf", TfidfVectorizer(**TFIDF_KWARGS)),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }
