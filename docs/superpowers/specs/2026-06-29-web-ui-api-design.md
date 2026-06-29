# Web UI + API for the Spam Classifier — Design

Date: 2026-06-29

## Goal

Give the existing CLI-only spam classifier a web interface and a JSON API,
without changing the existing ML code. The web layer reuses the trained
`SpamClassifier` pipeline and the `models/metrics.json` artifact.

## Non-goals (YAGNI)

- No database, persistence, or message history.
- No authentication or user accounts.
- No retraining from the UI (training stays in the CLI).
- No deployment infrastructure (Docker, cloud) — local server only.

## Architecture

A new `spam_classifier.api` module builds a **FastAPI** application. It:

- Loads a single `SpamClassifier` instance at startup (lazily tolerant of a
  missing model — see Error Handling).
- Reads `models/metrics.json` on demand.
- Serves a static single-page frontend from `src/spam_classifier/web/`.

The web layer sits on top of the existing package exactly as the CLI does. No
changes to `data.py`, `preprocessing.py`, `model.py`, `train.py`, or
`predict.py`.

```
src/spam_classifier/
  api.py          # FastAPI app: JSON endpoints + serves the page
  web/
    index.html    # single page
    style.css
    app.js        # fetch() calls to the API
```

## Endpoints

| Method | Path       | Request body             | Response |
|--------|------------|--------------------------|----------|
| GET    | `/`        | —                        | `index.html` |
| POST   | `/predict` | `{"messages": ["..."]}`  | `[{text, label, spam_probability}]` |
| GET    | `/metrics` | —                        | contents of `models/metrics.json` |
| GET    | `/health`  | —                        | `{"status": "ok", "model_loaded": bool}` |

`/predict` always takes an array, so a single message is just a one-element
array. Each result mirrors the existing `SpamClassifier.predict()` output
(`label`, `spam_probability`) plus the original `text` for display.

### Request/response shapes

- `POST /predict` request: `{"messages": ["free prize now", "see you at 5"]}`
- `POST /predict` response:
  ```json
  [
    {"text": "free prize now", "label": "SPAM", "spam_probability": 0.97},
    {"text": "see you at 5",   "label": "HAM",  "spam_probability": 0.02}
  ]
  ```
- `GET /metrics` response: the raw JSON from `models/metrics.json`, e.g.
  `{"best_model": "naive_bayes", "accuracy": 1.0, "spam_f1": 1.0,
  "spam_precision": 1.0, "spam_recall": 1.0}`.

## Frontend (single page, 4 features)

1. **Metrics panel** — small card at the top, populated from `GET /metrics` on
   page load: best model name and spam F1 / precision / recall.
2. **Single classify** — a textarea + "Classify" button. Result card shows the
   label (SPAM/HAM) and a **confidence bar** whose fill width = spam
   probability and color shifts green → red as probability rises.
3. **Batch classify** — a second textarea (one message per line) *and* a `.txt`
   file upload. Produces a results table: one row per message with its text,
   label, and a mini confidence bar.
4. All requests go through `app.js` via `fetch()`. A status banner surfaces
   network/server errors instead of failing silently.

The page is plain HTML/CSS/vanilla JS — no build step, no framework.

## Error handling

- **Model missing at startup:** the API still boots. `model_loaded` is `false`
  in `/health`. `POST /predict` returns HTTP **503** with a clear message
  ("Model not trained — run `spam-classify train` first"). The UI shows this
  inline in the status banner.
- **Empty / whitespace-only input:** `POST /predict` returns HTTP **400** with a
  descriptive message. The UI also disables the Classify button when input is
  empty.
- **`/metrics` when metrics.json is absent:** returns HTTP **404** with a clear
  message; the UI hides the metrics panel.
- **Frontend fetch failures:** caught and shown in the status banner.

## Testing

New `tests/test_api.py` using FastAPI's `TestClient`:

- `/predict` classifies an obvious spam message as SPAM and an obvious ham
  message as HAM.
- `/predict` with multiple messages returns one result per message in order.
- `/predict` with empty input → 400.
- `/predict` with no model loaded → 503 (using a dependency override or a
  classifier-less app instance).
- `/metrics` returns the metrics JSON.
- `/health` reports `model_loaded` correctly.

Existing tests remain untouched and passing.

## Packaging / running

- Add `fastapi` and `uvicorn[standard]` to `[project.dependencies]`.
- Add `httpx` to `[project.optional-dependencies].dev` (required by
  `TestClient`).
- Add a `spam-serve` console script that launches uvicorn against the app.
- Document `spam-serve` (and the equivalent `uvicorn` command) in the README.

## Pinned versions

Match the project's exact-pin style:

- `fastapi==0.115.0`
- `uvicorn[standard]==0.30.6`
- `httpx==0.27.2` (dev)

(Versions verified compatible with Python >=3.10 at implementation time; adjust
only if a conflict surfaces during install.)
