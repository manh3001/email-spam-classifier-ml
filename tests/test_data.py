import pandas as pd
import pytest

from spam_classifier.data import load_dataset

CSV_CONTENT = (
    "v1,v2,,,\n"
    "ham,Hello there,,,\n"
    'spam,"Win cash, now!",,,\n'
)


def _write_csv(tmp_path):
    p = tmp_path / "spam.csv"
    p.write_text(CSV_CONTENT, encoding="latin-1")
    return p


def test_loads_two_columns(tmp_path):
    df = load_dataset(_write_csv(tmp_path))
    assert list(df.columns) == ["label", "message"]


def test_maps_labels_to_ints(tmp_path):
    df = load_dataset(_write_csv(tmp_path))
    assert sorted(df["label"].tolist()) == [0, 1]


def test_drops_junk_columns(tmp_path):
    df = load_dataset(_write_csv(tmp_path))
    assert df.shape == (2, 2)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "nope.csv")


def test_missing_columns_raises(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="latin-1")
    with pytest.raises(ValueError):
        load_dataset(p)
