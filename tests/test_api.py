"""Unit and integration tests for FastAPI REST endpoints."""

import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.constants import CLASSES
from src.dataset import generate_synthetic_fingerprint


@pytest.fixture
def client():
    return TestClient(app)


def test_api_info(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert "title" in data
    assert "Parul University" in data["institution"]
    assert data["classes"] == list(CLASSES)
    assert len(data["authors"]) == 4
    assert len(data["professors"]) == 3


def test_api_samples(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    data = res.json()
    assert "samples" in data
    assert len(data["samples"]) == 3


def test_api_predict_preset(client):
    for c in CLASSES:
        res = client.post("/api/predict-preset", json={"pattern_type": c})
        assert res.status_code == 200
        data = res.json()
        assert data["predicted_class"] in CLASSES
        assert "confidence" in data
        assert "probabilities" in data
        assert "preprocessing_stages" in data


def test_api_predict_preset_invalid(client):
    res = client.post("/api/predict-preset", json={"pattern_type": "unknown_pattern"})
    assert res.status_code == 400


def test_api_predict_upload_image(client):
    # Create synthetic image bytes
    img = generate_synthetic_fingerprint("arch", size=64, seed=42)
    _, buffer = cv2.imencode(".png", img)
    file_bytes = io.BytesIO(buffer.tobytes())

    res = client.post(
        "/api/predict",
        files={"file": ("test_print.png", file_bytes, "image/png")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["predicted_class"] in CLASSES
    assert "preprocessing_stages" in data


def test_api_predict_upload_empty_file(client):
    empty_bytes = io.BytesIO(b"")
    res = client.post(
        "/api/predict",
        files={"file": ("empty.png", empty_bytes, "image/png")},
    )
    assert res.status_code == 400


def test_api_evaluation_report(client):
    res = client.get("/api/evaluation-report")
    assert res.status_code == 200
    data = res.json()
    assert "summary" in data
    assert "per_class" in data
    assert "confusion_matrix" in data
    assert "confusion_matrix_plot" in data


def test_api_serve_index(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "Automated Fingerprint Pattern Classification" in res.text
