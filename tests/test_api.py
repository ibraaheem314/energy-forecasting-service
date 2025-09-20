from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "healthy"

def test_forecast_minimal():
    # Test avec un horizon plus petit pour accélérer les tests
    r = client.post("/forecast", json={"horizon": 1, "city": "Paris", "with_intervals": True})
    assert r.status_code == 200
    body = r.json()
    assert "timestamps" in body and len(body["timestamps"]) == 1
    assert "yhat" in body and len(body["yhat"]) == 1