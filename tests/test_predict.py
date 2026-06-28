import joblib
import pytest

from spam_classifier.model import build_pipelines
from spam_classifier.predict import SpamClassifier


def _fitted_classifier():
    pipe = build_pipelines()["logistic_regression"]
    msgs = [
        "win free money now click here",
        "free cash prize claim now",
        "winner free voucher click",
        "are we still meeting tomorrow",
        "see you in class later",
        "thanks for the lift home",
    ]
    labels = [1, 1, 1, 0, 0, 0]
    pipe.fit(msgs, labels)
    return SpamClassifier(pipe)


def test_predict_returns_label_and_probability():
    clf = _fitted_classifier()
    out = clf.predict("free money click here")
    assert out["label"] in {"SPAM", "HAM"}
    assert 0.0 <= out["spam_probability"] <= 1.0


def test_predict_spam_message():
    clf = _fitted_classifier()
    assert clf.predict("free money prize claim now click")["label"] == "SPAM"


def test_predict_ham_message():
    clf = _fitted_classifier()
    assert clf.predict("see you in class tomorrow")["label"] == "HAM"


def test_predict_batch():
    clf = _fitted_classifier()
    out = clf.predict_batch(["free prize now", "see you later"])
    assert len(out) == 2
    assert all("label" in o and "spam_probability" in o for o in out)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        SpamClassifier.load(tmp_path / "missing.joblib")


def test_load_roundtrip(tmp_path):
    clf = _fitted_classifier()
    path = tmp_path / "m.joblib"
    joblib.dump(clf.pipeline, path)
    loaded = SpamClassifier.load(path)
    assert loaded.predict("free prize now")["label"] in {"SPAM", "HAM"}
