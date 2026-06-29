# Spam Classifier Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the flat-script SMS spam classifier into a clean, tested, installable Python CLI package.

**Architecture:** A `src/`-layout package `spam_classifier` with focused modules (config, data, preprocessing, model, train, predict, cli). Training fits and compares two scikit-learn `Pipeline`s (TF-IDF + NB, TF-IDF + LogReg) and persists the winner as a single `.joblib` artifact. A single `spam-classify` CLI drives train/predict.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, joblib, pytest.

## Global Constraints

- Python >= 3.10 (developed on 3.12).
- Pinned dependency versions in `requirements.txt`: pandas==2.2.2, numpy==1.26.4, scikit-learn==1.5.1, joblib==1.4.2.
- Dev deps in `requirements-dev.txt`: pytest==8.2.2, matplotlib==3.9.0, plus `-r requirements.txt`.
- Package lives under `src/spam_classifier/`. All imports use the `spam_classifier` package name.
- Persist exactly ONE artifact (the winning `Pipeline`) to `models/spam_pipeline.joblib`; write metrics to `models/metrics.json`.
- Label mapping: `ham -> 0`, `spam -> 1`. Spam is the positive class.
- Model selection metric: F1 on the spam (positive) class.
- The data file `data/spam.csv` is NOT edited; junk columns are dropped at load time.
- TDD: write the failing test first for every behavior-bearing module.

---

### Task 1: Project scaffolding & packaging

**Files:**
- Create: `pyproject.toml`
- Create: `src/spam_classifier/__init__.py`
- Create: `src/spam_classifier/config.py`
- Create: `requirements.txt` (overwrite existing)
- Create: `requirements-dev.txt`
- Modify: `.gitignore`
- Create: `tests/__init__.py`

**Interfaces:**
- Produces: package `spam_classifier` importable; `spam_classifier.config` exposing `DATA_PATH: Path`, `MODELS_DIR: Path`, `MODEL_PATH: Path`, `METRICS_PATH: Path`, `RANDOM_STATE: int = 42`, `TEST_SIZE: float = 0.2`, `TFIDF_KWARGS: dict`.

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "spam-classifier"
version = "0.1.0"
description = "SMS spam classifier (TF-IDF + scikit-learn)"
requires-python = ">=3.10"
dependencies = [
    "pandas==2.2.2",
    "numpy==1.26.4",
    "scikit-learn==1.5.1",
    "joblib==1.4.2",
]

[project.optional-dependencies]
dev = ["pytest==8.2.2", "matplotlib==3.9.0"]

[project.scripts]
spam-classify = "spam_classifier.cli:main"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 2: Create `src/spam_classifier/__init__.py`**

```python
"""SMS spam classifier package."""

__version__ = "0.1.0"
```

- [ ] **Step 3: Create `src/spam_classifier/config.py`**

```python
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
```

- [ ] **Step 4: Overwrite `requirements.txt`**

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.1
joblib==1.4.2
```

- [ ] **Step 5: Create `requirements-dev.txt`**

```
-r requirements.txt
pytest==8.2.2
matplotlib==3.9.0
```

- [ ] **Step 6: Modify `.gitignore`** — add `models/` under the model-files section.

```
# Model files
*.pkl
models/
```

- [ ] **Step 7: Create empty `tests/__init__.py`** (empty file).

- [ ] **Step 8: Verify package imports**

Run: `python -c "import sys; sys.path.insert(0,'src'); import spam_classifier.config as c; print(c.RANDOM_STATE, c.MODEL_PATH.name)"`
Expected: `42 spam_pipeline.joblib`

- [ ] **Step 9: Commit**

```bash
git add pyproject.toml src/spam_classifier/__init__.py src/spam_classifier/config.py requirements.txt requirements-dev.txt .gitignore tests/__init__.py
git commit -m "feat: scaffold spam_classifier package and config"
```

---

### Task 2: Text preprocessing

**Files:**
- Create: `src/spam_classifier/preprocessing.py`
- Test: `tests/test_preprocessing.py`

**Interfaces:**
- Produces: `clean_text(text: str) -> str` — lowercases, removes characters that are not `[a-z0-9 ]`, collapses surrounding whitespace.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_preprocessing.py
from spam_classifier.preprocessing import clean_text


def test_lowercases():
    assert clean_text("HELLO World") == "hello world"


def test_removes_punctuation_keeps_spaces():
    assert clean_text("Win!!! Free $$$ money") == "win free  money"


def test_keeps_alphanumeric():
    assert clean_text("Call 0800 now") == "call 0800 now"


def test_idempotent():
    once = clean_text("Hey, THERE!!")
    assert clean_text(once) == once
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_preprocessing.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.preprocessing'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/preprocessing.py
"""Text cleaning used before vectorization."""

import re

_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


def clean_text(text: str) -> str:
    """Lowercase and strip characters that are not letters, digits, or spaces."""
    text = str(text).lower()
    return _NON_ALNUM.sub("", text)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_preprocessing.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spam_classifier/preprocessing.py tests/test_preprocessing.py
git commit -m "feat: add clean_text preprocessing"
```

---

### Task 3: Dataset loading

**Files:**
- Create: `src/spam_classifier/data.py`
- Test: `tests/test_data.py`

**Interfaces:**
- Consumes: `spam_classifier.config.DATA_PATH`.
- Produces: `load_dataset(path: Path | str = DATA_PATH) -> pandas.DataFrame` with exactly two columns `label` (int: 0/1) and `message` (str). Drops junk columns, maps ham/spam. Raises `FileNotFoundError` if file missing; `ValueError` if `v1`/`v2` columns absent.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.data'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/data.py
"""Load and normalize the SMS spam dataset."""

from pathlib import Path

import pandas as pd

from .config import DATA_PATH

_LABEL_MAP = {"ham": 0, "spam": 1}


def load_dataset(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read spam.csv, keep label/message columns, map labels to 0/1."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    raw = pd.read_csv(path, encoding="latin-1")
    if not {"v1", "v2"}.issubset(raw.columns):
        raise ValueError(
            f"Expected columns 'v1' and 'v2' in {path}, got {list(raw.columns)}"
        )

    df = raw[["v1", "v2"]].rename(columns={"v1": "label", "v2": "message"})
    df = df.dropna(subset=["label", "message"])
    df["label"] = df["label"].str.strip().str.lower().map(_LABEL_MAP)
    if df["label"].isna().any():
        raise ValueError("Found label values outside {'ham','spam'}")
    df["label"] = df["label"].astype(int)
    df["message"] = df["message"].astype(str)
    return df.reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spam_classifier/data.py tests/test_data.py
git commit -m "feat: add load_dataset with junk-column handling"
```

---

### Task 4: Model pipelines

**Files:**
- Create: `src/spam_classifier/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: `spam_classifier.config.TFIDF_KWARGS`.
- Produces: `build_pipelines() -> dict[str, sklearn.pipeline.Pipeline]` with keys `"naive_bayes"` and `"logistic_regression"`. Each pipeline has steps named `"tfidf"` and `"clf"`. LogReg uses `class_weight="balanced"`, `max_iter=1000`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_model.py
from sklearn.pipeline import Pipeline

from spam_classifier.model import build_pipelines


def test_returns_both_pipelines():
    pipes = build_pipelines()
    assert set(pipes) == {"naive_bayes", "logistic_regression"}


def test_are_pipelines_with_expected_steps():
    for pipe in build_pipelines().values():
        assert isinstance(pipe, Pipeline)
        assert [name for name, _ in pipe.steps] == ["tfidf", "clf"]


def test_can_fit_and_predict():
    pipe = build_pipelines()["naive_bayes"]
    pipe.fit(["win free money", "see you tomorrow", "free cash now"], [1, 0, 1])
    assert pipe.predict(["free money"]).shape == (1,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.model'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/model.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_model.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spam_classifier/model.py tests/test_model.py
git commit -m "feat: add build_pipelines (NB + LogReg)"
```

---

### Task 5: Prediction (SpamClassifier)

**Files:**
- Create: `src/spam_classifier/predict.py`
- Test: `tests/test_predict.py`

**Interfaces:**
- Consumes: `clean_text` (Task 2), `build_pipelines` (Task 4), `config.MODEL_PATH`, joblib.
- Produces:
  - `class SpamClassifier` wrapping a fitted pipeline.
  - `SpamClassifier(pipeline)` constructor.
  - classmethod `SpamClassifier.load(path=MODEL_PATH) -> SpamClassifier` — raises `FileNotFoundError` with an actionable message if missing.
  - `.predict(text: str) -> dict` → `{"label": "SPAM"|"HAM", "spam_probability": float}`.
  - `.predict_batch(texts: list[str]) -> list[dict]`.
  - Predictions clean text via `clean_text` before inference.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_predict.py
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
    assert len(out) == 2 and all("label" in o for o in out)


def test_load_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="train"):
        SpamClassifier.load(tmp_path / "missing.joblib")


def test_load_roundtrip(tmp_path):
    clf = _fitted_classifier()
    path = tmp_path / "m.joblib"
    joblib.dump(clf.pipeline, path)
    loaded = SpamClassifier.load(path)
    assert loaded.predict("free prize now")["label"] in {"SPAM", "HAM"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_predict.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.predict'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/predict.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_predict.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spam_classifier/predict.py tests/test_predict.py
git commit -m "feat: add SpamClassifier predict with probability"
```

---

### Task 6: Training pipeline

**Files:**
- Create: `src/spam_classifier/train.py`
- Test: `tests/test_train.py`

**Interfaces:**
- Consumes: `load_dataset` (Task 3), `clean_text` (Task 2), `build_pipelines` (Task 4), config paths/params.
- Produces: `train(data_path=DATA_PATH, model_path=MODEL_PATH, metrics_path=METRICS_PATH, test_size=TEST_SIZE, random_state=RANDOM_STATE) -> dict`. The returned dict has keys `best_model` (str), `accuracy` (float), `spam_f1` (float). Side effects: writes the winning pipeline to `model_path` and a metrics summary JSON to `metrics_path` (creating parent dir). Selection by spam-class F1.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.train'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/train.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_train.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add src/spam_classifier/train.py tests/test_train.py
git commit -m "feat: add training pipeline with model comparison"
```

---

### Task 7: CLI

**Files:**
- Create: `src/spam_classifier/cli.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: `train` (Task 6), `SpamClassifier` (Task 5).
- Produces: `main(argv: list[str] | None = None) -> int`. Subcommands:
  - `train [--data PATH] [--model PATH] [--test-size F] [--seed N]`
  - `predict (MESSAGE | --file PATH) [--model PATH]`
  Returns 0 on success, non-zero on error. Console script `spam-classify` → `main`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'spam_classifier.cli'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/spam_classifier/cli.py
"""Command-line interface for training and prediction."""

import argparse
import logging
import sys
from pathlib import Path

from .config import MODEL_PATH
from .predict import SpamClassifier
from .train import train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam-classify")
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Train and persist the best model")
    t.add_argument("--data", default=None)
    t.add_argument("--model", default=str(MODEL_PATH))
    t.add_argument("--test-size", type=float, default=0.2)
    t.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("predict", help="Classify a message or a file of messages")
    p.add_argument("message", nargs="?", default=None)
    p.add_argument("--file", default=None)
    p.add_argument("--model", default=str(MODEL_PATH))
    return parser


def _run_train(args) -> int:
    from .config import DATA_PATH

    train(
        data_path=args.data or DATA_PATH,
        model_path=args.model,
        test_size=args.test_size,
        random_state=args.seed,
    )
    return 0


def _run_predict(args) -> int:
    try:
        clf = SpamClassifier.load(args.model)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.file:
        lines = [
            ln.strip()
            for ln in Path(args.file).read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
    elif args.message is not None:
        lines = [args.message]
    else:
        print("Error: provide a message or --file PATH", file=sys.stderr)
        return 2

    for text, result in zip(lines, clf.predict_batch(lines)):
        print(f"[{result['label']}] ({result['spam_probability']:.2f}) {text}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _build_parser().parse_args(argv)
    if args.command == "train":
        return _run_train(args)
    if args.command == "predict":
        return _run_predict(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_cli.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Run the full test suite**

Run: `pytest -v`
Expected: all tests pass (preprocessing, data, model, predict, train, cli).

- [ ] **Step 6: Commit**

```bash
git add src/spam_classifier/cli.py tests/test_cli.py
git commit -m "feat: add spam-classify CLI (train/predict)"
```

---

### Task 8: Remove legacy scripts, rewrite README, end-to-end check

**Files:**
- Delete: `train.py` (root)
- Delete: `predict.py` (root)
- Modify: `README.md`
- Modify: `test_messages.txt` (keep as sample input; no change needed unless tidying)

**Interfaces:**
- Consumes: working CLI from Task 7.
- Produces: no legacy duplicated scripts; accurate README.

- [ ] **Step 1: Delete legacy flat scripts**

```bash
git rm train.py predict.py
```

- [ ] **Step 2: Rewrite `README.md`** with this content:

```markdown
# Email / SMS Spam Classifier (Machine Learning)

Classifies SMS messages as **spam** or **ham** using TF-IDF features and
scikit-learn. Trains and compares Multinomial Naive Bayes and Logistic
Regression, then persists the better model (by spam-class F1) as a single
pipeline artifact.

## Project layout

```
src/spam_classifier/   # package: data, preprocessing, model, train, predict, cli
tests/                 # pytest suite
data/spam.csv          # dataset (UCI SMS Spam Collection format)
models/                # saved model + metrics (created by training; gitignored)
```

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e ".[dev]"         # installs the package + dev tools
```

## Usage

Train (writes `models/spam_pipeline.joblib` and `models/metrics.json`):

```bash
spam-classify train
```

Predict a single message:

```bash
spam-classify predict "Congratulations! You won a free iPhone, click now"
```

Predict a file of messages (one per line):

```bash
spam-classify predict --file test_messages.txt
```

## Testing

```bash
pytest
```

## How it works

1. `data.load_dataset` reads `spam.csv`, drops junk columns, maps `ham/spam` → `0/1`.
2. `preprocessing.clean_text` lowercases and strips non-alphanumeric characters.
3. `model.build_pipelines` builds TF-IDF + (NB | LogReg) pipelines.
4. `train.train` does a stratified split, fits both, and saves the best by spam F1.
5. `predict.SpamClassifier` loads the saved pipeline and classifies new messages.

Actual metrics are written to `models/metrics.json` after each training run.
```

- [ ] **Step 3: Install package in editable mode and run end-to-end**

Run:
```bash
pip install -e ".[dev]" && spam-classify train && spam-classify predict --file test_messages.txt
```
Expected: training prints reports for both models and a "Best model" line; prediction prints a `[SPAM]`/`[HAM]` line per message. `models/spam_pipeline.joblib` and `models/metrics.json` exist.

- [ ] **Step 4: Run full test suite once more**

Run: `pytest -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove legacy scripts and rewrite README"
```

---

## Self-Review notes

- Spec coverage: config (T1), data junk-column handling (T3), clean_text (T2), model comparison NB+LogReg (T4/T6), stratified split + spam-F1 selection (T6), single artifact persistence (T6), actionable missing-model error (T5), CLI single+batch+probability (T5/T7), tests for all behavior modules (T2-T7), README rewrite + .gitignore + legacy removal (T1/T8), pinned deps (T1). All covered.
- Type consistency: `build_pipelines` keys `naive_bayes`/`logistic_regression` used consistently in T4/T6; pipeline step names `tfidf`/`clf` consistent; `SpamClassifier.pipeline`, `.load`, `.predict`, `.predict_batch` consistent across T5/T6/T7; `train(...)` signature consistent across T6/T7 (note: CLI omits `metrics_path`, relying on its default — intentional).
- No placeholders: all steps contain concrete code/commands.
```
