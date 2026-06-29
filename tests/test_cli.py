from spam_classifier.cli import main


def _make_csv(tmp_path):
    rows = ["v1,v2,,,"]
    for _ in range(15):
        rows.append('spam,"free money prize winner click now claim cash",,,')
        rows.append('ham,"hey are we meeting for lunch tomorrow at noon",,,')
    csv = tmp_path / "spam.csv"
    csv.write_text("\n".join(rows) + "\n", encoding="latin-1")
    return csv


def test_train_then_predict_message(tmp_path, capsys):
    csv = _make_csv(tmp_path)
    model = tmp_path / "m.joblib"
    rc = main(["train", "--data", str(csv), "--model", str(model),
               "--test-size", "0.4", "--seed", "0"])
    assert rc == 0 and model.exists()

    rc = main(["predict", "free money prize claim now", "--model", str(model)])
    assert rc == 0
    assert "SPAM" in capsys.readouterr().out


def test_predict_from_file(tmp_path, capsys):
    csv = _make_csv(tmp_path)
    model = tmp_path / "m.joblib"
    main(["train", "--data", str(csv), "--model", str(model),
          "--test-size", "0.4", "--seed", "0"])
    msgs = tmp_path / "msgs.txt"
    msgs.write_text("free prize now\nsee you tomorrow\n", encoding="utf-8")
    rc = main(["predict", "--file", str(msgs), "--model", str(model)])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.count("SPAM") + out.count("HAM") == 2


def test_predict_without_model_errors(tmp_path, capsys):
    rc = main(["predict", "hello", "--model", str(tmp_path / "none.joblib")])
    assert rc != 0
    assert "Error" in capsys.readouterr().err
