from __future__ import annotations

from src.classifier import CATEGORY_LABELS, WorkOrderClassifier


def test_onnx_classifier_roundtrip_shape() -> None:
    clf = WorkOrderClassifier(mode="onnx").load()

    result = clf.classify(
        "Responded to pump P-104 high vibration. Found bearing wear and replaced "
        "mechanical seal. Returned unit to service after alignment."
    )

    assert result["category"] in CATEGORY_LABELS
    assert set(result["all_scores"]) == set(CATEGORY_LABELS)
    assert all(0 <= score <= 1 for score in result["all_scores"].values())
    assert result["confidence"] == max(result["all_scores"].values())
    assert result["model_used"] == "distilbert_lora_int8"
