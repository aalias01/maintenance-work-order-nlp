from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app
from src.classifier import CATEGORY_LABELS


def test_health_contract_includes_extractor_mode() -> None:
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["classifier_loaded"] is True
    assert data["embeddings_loaded"] is True
    assert data["model_mode"]
    assert data["extractor_mode"] == "rule_based"
    assert data["corpus_size"] > 0


def test_classify_contract_shape() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/classify",
            json={
                "text": (
                    "Motor M-210 tripped on overload. Found shorted heater element "
                    "and replaced wiring before returning panel to service."
                )
            },
        )

    assert response.status_code == 200
    data = response.json()
    assert data["category"] in CATEGORY_LABELS
    assert set(data["all_scores"]) == set(CATEGORY_LABELS)
    assert data["confidence"] == max(data["all_scores"].values())
    assert data["model_used"]
    assert isinstance(data["extracted_fields"], dict)


def test_classify_rejects_schema_bounds() -> None:
    with TestClient(app) as client:
        short_response = client.post("/classify", json={"text": "short"})
        long_response = client.post("/classify", json={"text": "x" * 2001})

    assert short_response.status_code == 422
    assert long_response.status_code == 422
