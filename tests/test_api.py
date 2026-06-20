from fastapi.testclient import TestClient
from app.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "Api up and running..."


def test_predict():
    payload = {"review": "I love this movie"}

    response = client.post("/predict", json=payload)
    data = response.json()

    assert response.status_code == 200
    assert "sentiment" in data
    assert "prediction" in data
    assert "probability" in data
