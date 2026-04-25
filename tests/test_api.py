from __future__ import annotations

from fastapi.testclient import TestClient

from deepdive.api.main import app


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "version" in body
