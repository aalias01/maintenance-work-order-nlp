from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data" / "work_orders.csv"
OUT_PATH = ROOT / "models" / "corpus_meta.json"
INDEX_DIRS = (ROOT / "models" / "onnx", ROOT / "models")
REQUIRED_COLUMNS = ("work_order_id", "failure_category", "text")


def _load_embedding_reference() -> tuple[np.ndarray, list[str], Path]:
    for base in INDEX_DIRS:
        index_path = base / "embeddings_index.npy"
        texts_path = base / "embeddings_texts.json"
        if index_path.exists() and texts_path.exists():
            embeddings = np.load(index_path)
            texts = json.loads(texts_path.read_text())
            return embeddings, texts, base
    searched = ", ".join(str(path) for path in INDEX_DIRS)
    raise FileNotFoundError(f"No embeddings index and texts found in: {searched}")


def _row_text(value: object) -> str:
    return str(value)[:300]


def build_meta() -> list[dict[str, str]]:
    if not DATA_CSV.exists():
        raise FileNotFoundError(
            "data/work_orders.csv is missing. Regenerate it with "
            "`python data/synthetic_generator.py`, then rerun this script."
        )

    df = pd.read_csv(DATA_CSV)
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"data/work_orders.csv is missing required columns: {missing}")

    embeddings, embedding_texts, index_dir = _load_embedding_reference()
    meta = [
        {
            "work_order_id": str(row.work_order_id),
            "failure_category": str(row.failure_category),
            "text": _row_text(row.text),
        }
        for row in df.itertuples(index=False)
    ]

    if len(meta) != embeddings.shape[0]:
        raise AssertionError(
            f"metadata rows ({len(meta)}) do not match embeddings rows ({embeddings.shape[0]})"
        )
    if len(embedding_texts) != embeddings.shape[0]:
        raise AssertionError(
            "embeddings_texts.json length "
            f"({len(embedding_texts)}) does not match embeddings rows ({embeddings.shape[0]})"
        )

    spot_indices = (0, len(meta) // 2, len(meta) - 1)
    for idx in spot_indices:
        meta_text = meta[idx]["text"]
        embedded_text = str(embedding_texts[idx])
        if not embedded_text.startswith(meta_text):
            raise AssertionError(
                f"alignment failed at row {idx}: corpus metadata text does not match {index_dir}"
            )

    return meta


def main() -> None:
    meta = build_meta()
    OUT_PATH.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)} with {len(meta)} aligned rows.")


if __name__ == "__main__":
    main()
