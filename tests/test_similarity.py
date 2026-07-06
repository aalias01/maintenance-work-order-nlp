from __future__ import annotations

import numpy as np

from src.nlp_pipeline import cosine_similarity_search


def test_cosine_similarity_search_returns_nearest_indices_in_order() -> None:
    query = np.array([1.0, 0.0], dtype=np.float32)
    corpus = np.array(
        [
            [0.0, 1.0],
            [0.9, 0.1],
            [1.0, 0.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )

    hits = cosine_similarity_search(query, corpus, top_k=3)

    assert [idx for idx, _score in hits] == [2, 1, 0]
    assert hits[0][1] > hits[1][1] > hits[2][1]
