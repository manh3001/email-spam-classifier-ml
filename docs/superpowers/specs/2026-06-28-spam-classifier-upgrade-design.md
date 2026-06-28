# Email Spam Classifier — Upgrade Design

**Date:** 2026-06-28
**Status:** Approved

## Goal

Upgrade the existing flat-script SMS spam classifier into a clean, installable,
well-tested Python CLI project that follows professional standards. No web/API
layer (explicitly out of scope).

## Current state (problems being fixed)

- `README.md` is broken: duplicate/contradictory accuracy numbers (95% vs 96.7%)
  and an unclosed code fence.
- `predict.py` crashes with a cryptic `FileNotFoundError` if run before
  `train.py` (no model artifact exists).
- `train.py` and `predict.py` save/load **two** loose `.pkl` files
  (`spam_model.pkl`, `tfidf_vectorizer.pkl`) that can drift out of sync.
- `data/spam.csv` carries 3 junk empty trailing columns.
- README advertises "Naive Bayes, Logistic Regression" but only Naive Bayes is
  implemented; no model comparison.
- Train/test split is not stratified despite class imbalance (~13% spam).
- No tests, no error handling, no logging, no package structure.

## Target architecture

Installable package under `src/`, driven by a single CLI. Core idea: persist a
scikit-learn `Pipeline` (TF-IDF vectorizer + classifier) as **one** artifact so
the vectorizer and model can never drift apart.

```
email-spam-classifier-ml/
├── data/spam.csv
├── models/                      # saved artifacts (gitignored)
├── src/spam_classifier/
│   ├── __init__.py
│   ├── config.py                # paths, constants, default params
│   ├── data.py                  # load_dataset()
│   ├── preprocessing.py         # clean_text()
│   ├── model.py                 # build_pipelines() -> NB + LogReg
│   ├── train.py                 # train, compare, evaluate, persist best
│   ├── predict.py               # SpamClassifier: load + predict
│   └── cli.py                   # `train` and `predict` subcommands
├── tests/
├── pyproject.toml               # metadata + console entry point
├── requirements.txt
├── requirements-dev.txt
├── README.md
└── .gitignore
```

## Components

### `config.py`
Centralizes paths (data file, models dir, artifact path, metrics path) and
default hyperparameters (test_size=0.2, random_state=42, tfidf params).

### `data.py` — `load_dataset(path) -> pd.DataFrame`
Reads `spam.csv` with `latin-1` encoding, selects only the `v1`/`v2` columns
(ignoring junk trailing columns), renames to `label`/`message`, maps
`ham->0, spam->1`. Raises a clear error if the file or expected columns are
missing.

### `preprocessing.py` — `clean_text(text) -> str`
Lowercases and strips non-alphanumeric characters (preserving spaces). Pure
function, easy to unit test.

### `model.py` — `build_pipelines() -> dict[str, Pipeline]`
Returns named sklearn `Pipeline`s: `"naive_bayes"` (TfidfVectorizer +
MultinomialNB) and `"logistic_regression"` (TfidfVectorizer + LogisticRegression
with `class_weight="balanced"`). TF-IDF uses `stop_words="english"`. Text is
cleaned before vectorization (cleaning applied in the training/predict flow).

### `train.py` — `train(...)`
1. Load data, clean messages.
2. Stratified train/test split.
3. Fit both pipelines.
4. Evaluate each: accuracy + full classification report; select winner by
   **F1 score on the spam class** (appropriate for the imbalance).
5. Print reports for both models and announce the winner.
6. Persist winning pipeline to `models/spam_pipeline.joblib` and a `metrics.json`
   summary (model name, accuracy, spam precision/recall/F1).

### `predict.py` — `class SpamClassifier`
- `SpamClassifier.load(path=default)` — loads the pipeline; raises a clear,
  actionable error ("Model not found — run `spam-classify train` first") if the
  artifact is missing.
- `.predict(text) -> {"label": "SPAM"|"HAM", "spam_probability": float}`
- `.predict_batch(texts) -> list[...]`
Applies the same `clean_text` as training before inference.

### `cli.py`
`argparse` with two subcommands:
- `spam-classify train [--data PATH] [--test-size F] [--seed N]`
- `spam-classify predict ("message" | --file PATH)`
Console entry point `spam-classify` declared in `pyproject.toml`.

## Data flow

`load_dataset` → `clean_text` → `Pipeline(TfidfVectorizer → classifier)` →
persisted pipeline → `SpamClassifier.predict`.

## Error handling

- Missing/invalid data file → clear `FileNotFoundError`/`ValueError` with the
  offending path.
- Missing model artifact at predict time → actionable message telling the user
  to train first.
- Empty/whitespace prediction input → handled gracefully (still classified, no
  crash).
- Logging via the `logging` module for the training pipeline (INFO level
  progress), replacing scattered `print` calls in the training flow.

## Testing (TDD, written first)

- `test_preprocessing.py` — `clean_text` lowercasing, punctuation removal, space
  preservation, idempotency.
- `test_data.py` — `load_dataset` drops junk columns, renames, maps labels;
  errors on missing file. Uses a tiny temp CSV fixture.
- `test_predict.py` — `SpamClassifier` predicts on a small in-test-trained
  pipeline; raises the actionable error when artifact missing.
- `test_train.py` — fast smoke test: training on a small slice produces an
  artifact and a metrics file, winner selection runs end to end.

Run with `pytest`. Target: all tests green, no network/large-data dependence in
unit tests.

## Dependencies

- `requirements.txt` — pinned runtime deps (pandas, numpy, scikit-learn,
  joblib). Drop `matplotlib`/`nltk` from runtime (unused by core); matplotlib
  stays only if the notebook keeps the plot — moved to dev.
- `requirements-dev.txt` — pytest, matplotlib (for notebook), plus runtime via
  `-r requirements.txt`.
- Versions stay pinned.

## Out of scope (YAGNI)

- REST API / FastAPI, Docker, CI workflow.
- Deep-learning models, hyperparameter search.
- Persisting both NB and LogReg (only the winner is saved).

## Cleanups bundled with this work

- Rewrite `README.md` with accurate, single set of metrics and correct run
  instructions.
- `.gitignore` add `models/` (keep `*.pkl` ignore for back-compat).
- Junk CSV columns handled at load time — the data file itself is not edited.
