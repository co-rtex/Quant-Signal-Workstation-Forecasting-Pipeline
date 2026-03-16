"""API smoke tests."""

from fastapi.testclient import TestClient

from quant_signal.api.app import app


def test_health_endpoints() -> None:
    """Health endpoints should respond successfully."""

    client = TestClient(app)

    live = client.get("/health/live")
    ready = client.get("/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert live.json()["status"] == "ok"
    assert ready.json()["status"] == "ready"
