"""FastAPI backend serving the spam classifier UI and JSON API."""

from pathlib import Path

from fastapi import FastAPI

from .config import METRICS_PATH, MODEL_PATH

WEB_DIR = Path(__file__).resolve().parent / "web"


def create_app(classifier=None, model_path=MODEL_PATH, metrics_path=METRICS_PATH) -> FastAPI:
    app = FastAPI(title="Spam Classifier")
    return app


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port)
