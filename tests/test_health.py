import os
import pytest


if not os.getenv("RUN_DB_TESTS"):
    pytest.skip("Skip API tests: set RUN_DB_TESTS=1 with a running Postgres", allow_module_level=True)


def test_health_endpoint():
    """
    Basic health check. Requires the app to start with a real DATABASE_URL.
    """
    from fastapi.testclient import TestClient
    from app.main import app

    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json().get("status") == "ok"

