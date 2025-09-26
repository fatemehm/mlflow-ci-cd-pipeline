import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

@pytest.mark.asyncio
async def test_predict_positive():
    response = client.post("/predict", json={"review": "This movie is great!"})
    assert response.status_code == 200
    data = response.json()
    assert data["review"] == "This movie is great!"
    assert data["predicted_sentiment"] == "positive"
    assert 0.95 <= data["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_predict_negative():
    response = client.post("/predict", json={"review": "This product is terrible."})
    assert response.status_code == 200
    data = response.json()
    assert data["review"] == "This product is terrible."
    assert data["predicted_sentiment"] == "negative"
    assert 0.95 <= data["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_predict_uncertain_mixed():
    response = client.post("/predict", json={"review": "Great design but poor quality."})
    assert response.status_code == 200
    data = response.json()
    assert data["review"] == "Great design but poor quality."
    assert data["predicted_sentiment"] == "uncertain"
    assert 0.85 <= data["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_predict_uncertain_neutral():
    response = client.post("/predict", json={"review": "Okay product."})
    assert response.status_code == 200
    data = response.json()
    assert data["review"] == "Okay product."
    assert data["predicted_sentiment"] == "uncertain"
    assert 0.95 <= data["confidence"] <= 1.0

@pytest.mark.asyncio
async def test_predict_empty():
    response = client.post("/predict", json={"review": ""})
    assert response.status_code == 400
    assert response.json() == {"detail": "Review text is empty after cleaning"}

@pytest.mark.asyncio
async def test_predict_malformed_json():
    response = client.post("/predict", json={"wrong_key": "This movie is great!"})
    assert response.status_code == 422  # Unprocessable Entity
    assert "review" in response.json()["detail"][0]["loc"]

@pytest.mark.asyncio
async def test_predict_batch():
    response = client.post("/predict_batch", json={
        "reviews": ["This movie is great!", "Okay product.", "This product is terrible.", ""]
    })
    assert response.status_code == 200
    results = response.json()
    assert len(results) == 4
    assert results[0]["predicted_sentiment"] == "positive"
    assert results[1]["predicted_sentiment"] == "uncertain"
    assert results[2]["predicted_sentiment"] == "negative"
    assert results[3]["error"] == "Review text is empty after cleaning"