"""FastAPI backend serving the spam classifier UI and JSON API."""

from pathlib import Path

from fastapi import FastAPI

from .config import METRICS_PATH, MODEL_PATH
from .predict import SpamClassifier

WEB_DIR = Path(__file__).resolve().parent / "web"


def _load_classifier(model_path):
    try:
        return SpamClassifier.load(model_path)
    except FileNotFoundError:
        return None


def create_app(classifier=None, model_path=MODEL_PATH, metrics_path=METRICS_PATH) -> FastAPI:
    app = FastAPI(title="Spam Classifier")
    app.state.classifier = classifier if classifier is not None else _load_classifier(model_path)
    app.state.metrics_path = Path(metrics_path)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_loaded": app.state.classifier is not None}

    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
