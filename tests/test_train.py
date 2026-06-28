import json

from spam_classifier.train import train


def test_train_writes_artifacts_and_returns_summary(tmp_path):
    # Build a tiny but learnable dataset CSV in the v1,v2 format.
    rows = ["v1,v2,,,"]
    for _ in range(15):
        rows.append('spam,"free money prize winner click now claim cash",,,')
        rows.append('ham,"hey are we meeting for lunch tomorrow at noon",,,')
    csv = tmp_path / "spam.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="latin-1")

    model_path = tmp_path / "models" / "pipe.joblib"
    metrics_path = tmp_path / "models" / "metrics.json"

    summary = train(
        data_path=csv,
        model_path=model_path,
        metrics_path=metrics_path,
        test_size=0.4,
        random_state=0,
    )

    assert summary["best_model"] in {"naive_bayes", "logistic_regression"}
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert 0.0 <= summary["spam_f1"] <= 1.0
    assert model_path.exists()
    assert metrics_path.exists()
    saved = json.loads(metrics_path.read_text())
    assert saved["best_model"] == summary["best_model"]


def test_trained_model_is_usable(tmp_path):
    rows = ["v1,v2,,,"]
    for _ in range(15):
        rows.append('spam,"free money prize winner click now claim cash",,,')
        rows.append('ham,"hey are we meeting for lunch tomorrow at noon",,,')
    csv = tmp_path / "spam.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="latin-1")
    model_path = tmp_path / "m.joblib"

    train(data_path=csv, model_path=model_path,
          metrics_path=tmp_path / "metrics.json", test_size=0.4, random_state=0)

    from spam_classifier.predict import SpamClassifier
    clf = SpamClassifier.load(model_path)
    assert clf.predict("free money prize claim now")["label"] == "SPAM"
