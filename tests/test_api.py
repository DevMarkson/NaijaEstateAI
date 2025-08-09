import json
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from config import BEST_MODEL_PATH, METRICS_PATH, DATA_PATH


@pytest.fixture(scope="session", autouse=True)
def ensure_trained_model():
    """Train model once for API tests if artifacts missing.

    Keeps runtime low by skipping if model + metrics already exist.
    """
    if BEST_MODEL_PATH.exists() and METRICS_PATH.exists():
        return
    cmd = [
        sys.executable, "train.py",
        "--data", str(DATA_PATH),
        "--test-size", "0.25",
        "--skip-xgb",
        "--cv-folds", "1",  # speed: disable CV for this quick training
        "--max-rows", "4000"  # cap rows for faster CI
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Training failed for API tests: {res.stderr}\n{res.stdout}"
    assert BEST_MODEL_PATH.exists(), "Best model not created during setup"
    assert METRICS_PATH.exists(), "Metrics file not created during setup"


@pytest.fixture(scope="session")
def client():
    # Import after training so api_app can load artifacts
    from api_app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json().get("status") == "ok"


def test_model_info(client):
    r = client.get("/model/info")
    assert r.status_code == 200
    data = r.json()
    # Basic keys present
    for key in ["best", "rows", "test_size", "target_transform"]:
        assert key in data


def test_predict_success(client):
    payload = {
        "bedrooms": 3,
        "bathrooms": 3,
        "toilets": 4,
        "Serviced": 1,
        "Newly_Built": 0,
        "Furnished": 0,
        "property_type": "Apartment",
        "City": "Lagos",
        "Neighborhood": "Lekki"
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert "prediction" in data and isinstance(data["prediction"], (int, float))
    assert data["rounded"] >= 0


def test_predict_validation_error(client):
    # Missing a required field (Neighborhood) should 422
    bad_payload = {
        "bedrooms": 2,
        "bathrooms": 2,
        "toilets": 2,
        "Serviced": 0,
        "Newly_Built": 0,
        "Furnished": 0,
        "property_type": "Apartment",
        "City": "Lagos"
    }
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422
