from __future__ import annotations

import api.predictor as predictor


def test_corpus_meta_fallback_supplies_size_and_rows(monkeypatch) -> None:
    meta = [
        {
            "work_order_id": "WO-1",
            "text": "Pump P-104 bearing replaced.",
            "failure_category": "mechanical_failure",
        },
        {
            "work_order_id": "WO-2",
            "text": "Panel breaker tripped on shorted heater.",
            "failure_category": "electrical_failure",
        },
    ]
    monkeypatch.setattr(predictor, "_corpus_df", None)
    monkeypatch.setattr(predictor, "_corpus_meta", meta)

    assert predictor._corpus_size() == 2
    assert predictor._corpus_row(1) == meta[1]
