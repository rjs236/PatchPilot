"""Unit and integration tests for FastAPI service endpoints."""

import pytest
from fastapi.testclient import TestClient
from patchpilot.api import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_review_endpoint_valid_code():
    payload = {
        "code": "def f(items=[]):\n    pass\n",
        "filename": "test.py",
        "use_llm": False,
    }
    response = client.post("/review", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["filename"] == "test.py"
    assert "summary" in data
    assert "findings" in data
    assert data["summary"]["total"] >= 1


def test_review_endpoint_payload_exceeding_size_limit():
    huge_code = "a = 1\n" * 35000  # > 210,000 characters
    assert len(huge_code) > 200000

    payload = {
        "code": huge_code,
        "filename": "huge.py",
        "use_llm": False,
    }
    response = client.post("/review", json=payload)
    # FastAPI returns 422 Unprocessable Entity for schema validation or 400 for check
    assert response.status_code in (400, 422)


def test_review_endpoint_syntax_error():
    payload = {
        "code": "def broken(\n",
        "filename": "syntax.py",
        "use_llm": False,
    }
    response = client.post("/review", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["summary"]["high"] >= 1
    assert any(f["rule_id"] == "syntax-error" for f in data["findings"])
