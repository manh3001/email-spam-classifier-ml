# Email / SMS Spam Classifier (Machine Learning)

Classifies SMS messages as **spam** or **ham** using TF-IDF features and
scikit-learn. Trains and compares Multinomial Naive Bayes and Logistic
Regression, then persists the better model (by spam-class F1) as a single
pipeline artifact.

## Project layout

```text
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
