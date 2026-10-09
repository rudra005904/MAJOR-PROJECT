"""Tests for database integration (SQLite and PostgreSQL)."""

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.database import (
    BiometricExamination,
    get_db_stats,
    get_examination_by_id,
    init_db,
    list_examinations,
    log_examination,
)

client = TestClient(app)


def test_database_initialization():
    """Verifies that database schema initializes cleanly."""
    init_db()
    stats = get_db_stats()
    assert "total_examinations" in stats
    assert "database_type" in stats


def test_log_and_retrieve_examination():
    """Verifies logging a new examination and retrieving it."""
    init_db()
    data = {
        "filename": "unit_test_scan.bmp",
        "file_type": "BMP",
        "filesize_formatted": "64.0 KB",
        "predicted_class": "whorl",
        "predicted_class_title": "Whorl Pattern",
        "confidence_percentage": "94.20%",
        "probabilities": {"arch": 0.02, "loop": 0.038, "whorl": 0.942},
        "borderline_analysis": {"is_borderline": False},
        "latency_ms": 42.5,
        "source": "unit_test",
    }
    rec = log_examination(data)
    assert rec.examination_id.startswith("PU-BIO-")
    assert rec.predicted_class == "whorl"

    # Retrieve by exam ID
    fetched = get_examination_by_id(rec.examination_id)
    assert fetched is not None
    assert fetched["filename"] == "unit_test_scan.bmp"
    assert fetched["predicted_class"] == "whorl"
    assert fetched["confidence_percentage"] == "94.20%"


def test_list_examinations_filtering():
    """Verifies listing and filtering examinations by pattern."""
    result = list_examinations(limit=10, pattern_filter="whorl")
    assert "records" in result
    assert "total" in result
    for r in result["records"]:
        assert r["predicted_class"] == "whorl"


def test_api_examinations_endpoints():
    """Verifies GET /api/examinations and GET /api/database-stats."""
    res_stats = client.get("/api/database-stats")
    assert res_stats.status_code == 200
    assert "total_examinations" in res_stats.json()

    res_list = client.get("/api/examinations?limit=5")
    assert res_list.status_code == 200
    data = res_list.json()
    assert "records" in data
    assert "total" in data
