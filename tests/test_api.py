import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import unittest
from fastapi.testclient import TestClient
from app import app
import time

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "message": "Welcome to the Sentiment Analysis API. Use /predict for single predictions, /predict/batch for batch predictions, or /visualize for charts."
        })

    def test_predict_endpoint_post(self):
        response = self.client.post("/predict", json={"review": "This movie was amazing!"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["review"], "This movie was amazing!")
        self.assertEqual(data["cleaned_text"], "this movie was amazing")
        self.assertEqual(data["predicted_label"], 1)
        self.assertTrue(0 <= data["prediction_score"] <= 1)
        self.assertIsNone(data["error"])

    def test_predict_endpoint_get(self):
        response = self.client.get("/predict?review=This%20movie%20was%20amazing!")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["review"], "This movie was amazing!")
        self.assertEqual(data["cleaned_text"], "this movie was amazing")
        self.assertEqual(data["predicted_label"], 1)
        self.assertTrue(0 <= data["prediction_score"] <= 1)
        self.assertIsNone(data["error"])

    def test_predict_batch_endpoint(self):
        response = self.client.post("/predict/batch", json={
            "reviews": ["This movie was amazing!", "Terrible film, waste of time."]
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data["predictions"]), 2)
        self.assertEqual(data["predictions"][0]["review"], "This movie was amazing!")
        self.assertEqual(data["predictions"][0]["cleaned_text"], "this movie was amazing")
        self.assertEqual(data["predictions"][0]["predicted_label"], 1)
        self.assertEqual(data["predictions"][1]["review"], "Terrible film, waste of time.")
        self.assertEqual(data["predictions"][1]["cleaned_text"], "terrible film waste of time")
        self.assertEqual(data["predictions"][1]["predicted_label"], 0)

    def test_visualize_endpoint(self):
        response = self.client.get("/visualize")
        self.assertEqual(response.status_code, 200)
        self.assertIn("<!DOCTYPE html>", response.text)
        self.assertIn('<img src="/static/sentiment_distribution.png"', response.text)

if __name__ == "__main__":
    unittest.main()