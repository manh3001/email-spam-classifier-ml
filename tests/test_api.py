from fastapi.testclient import TestClient

from spam_classifier.api import create_app


def test_app_constructs():
    app = create_app()
    assert app is not None
    # TestClient context triggers startup/shutdown without error.
    with TestClient(app):
        pass
