from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

META_PATH = Path("models/corpus_meta.json")
INDEX_PATH = Path("models/embeddings_index.npy")
TEXTS_PATH = Path("models/embeddings_texts.json")


@pytest.mark.skipif(not META_PATH.exists(), reason="corpus metadata has not been generated")
def test_corpus_meta_aligns_with_embeddings_index() -> None:
    meta = json.loads(META_PATH.read_text())
    embeddings = np.load(INDEX_PATH)
    embedding_texts = json.loads(TEXTS_PATH.read_text())

    assert len(meta) == embeddings.shape[0]
    assert len(embedding_texts) == embeddings.shape[0]

    for idx in (0, len(meta) // 2, len(meta) - 1):
        assert embedding_texts[idx].startswith(meta[idx]["text"])
        assert meta[idx]["work_order_id"]
        assert meta[idx]["failure_category"]
