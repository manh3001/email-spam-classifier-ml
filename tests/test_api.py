from fastapi.testclient import TestClient

from spam_classifier.api import create_app
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
    pipe.fit(msgs, [1, 1, 1, 0, 0, 0])
    return SpamClassifier(pipe)


def test_app_constructs():
    app = create_app()
    assert app is not None
    # TestClient context triggers startup/shutdown without error.
    with TestClient(app):
        pass


def test_health_model_loaded():
    client = TestClient(create_app(classifier=_fitted_classifier()))
    body = client.get("/health").json()
    assert body == {"status": "ok", "model_loaded": True}


def test_health_no_model(tmp_path):
    client = TestClient(create_app(model_path=tmp_path / "none.joblib"))
    body = client.get("/health").json()
    assert body["model_loaded"] is False
