import json

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


def test_predict_single_spam():
    client = TestClient(create_app(classifier=_fitted_classifier()))
    r = client.post("/predict", json={"messages": ["free money prize claim now click"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["label"] == "SPAM"
    assert data[0]["text"] == "free money prize claim now click"
    assert 0.0 <= data[0]["spam_probability"] <= 1.0


def test_predict_batch_order():
    client = TestClient(create_app(classifier=_fitted_classifier()))
    r = client.post("/predict", json={"messages": ["free prize now", "see you in class tomorrow"]})
    data = r.json()
    assert [d["text"] for d in data] == ["free prize now", "see you in class tomorrow"]
    assert data[1]["label"] == "HAM"


def test_predict_empty_is_400():
    client = TestClient(create_app(classifier=_fitted_classifier()))
    assert client.post("/predict", json={"messages": []}).status_code == 400
    assert client.post("/predict", json={"messages": ["   "]}).status_code == 400


def test_predict_no_model_is_503(tmp_path):
    client = TestClient(create_app(model_path=tmp_path / "none.joblib"))
    r = client.post("/predict", json={"messages": ["hello"]})
    assert r.status_code == 503


def test_metrics_returns_json(tmp_path):
    mpath = tmp_path / "metrics.json"
    mpath.write_text(json.dumps({"best_model": "naive_bayes", "spam_f1": 1.0}), encoding="utf-8")
    client = TestClient(create_app(classifier=_fitted_classifier(), metrics_path=mpath))
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.json()["best_model"] == "naive_bayes"


def test_metrics_missing_is_404(tmp_path):
    client = TestClient(create_app(classifier=_fitted_classifier(), metrics_path=tmp_path / "none.json"))
    assert client.get("/metrics").status_code == 404


def test_root_serves_html():
    client = TestClient(create_app(classifier=_fitted_classifier()))
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Spam Classifier" in r.text
