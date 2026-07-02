from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.classifier import WorkOrderClassifier
from src.nlp_pipeline import OnnxEmbedder, cosine_similarity_search
from src.etl_extractor import ETLExtractor
from api.schemas import ClassifyRequest, ClassifyResponse, SimilarCase

MODEL_DIR = Path("models")
DATA_DIR = Path("data")
ONNX_DIR = MODEL_DIR / "onnx"

# ── Production serving = int8 ONNX ──────────────────────────────────────────
# The real DistilBERT+LoRA classifier and the MiniLM similarity embedder are
# served as int8 ONNX models via onnxruntime (no torch). Under PyTorch they need
# ~650 MB and OOM on Render's 512 MB free tier; as int8 ONNX the pair runs in
# ~250-300 MB, so the *real* fine-tuned model serves live. Artifacts are built
# once by scripts/build_onnx.py and committed under models/onnx/.
#
# If the ONNX artifacts are absent (e.g. before the first build), the classifier
# degrades to the small TF-IDF baseline so the endpoint still responds.

_clf: Optional[WorkOrderClassifier] = None
_embedder = None                       # OnnxEmbedder or sentence-transformers fallback
_embeddings: Optional[np.ndarray] = None
_corpus_texts: Optional[list[str]] = None
_corpus_df: Optional[pd.DataFrame] = None
_etl: Optional[ETLExtractor] = None


class _STEmbedder:
    """Local-dev fallback: sentence-transformers embedder (requires torch)."""
    def __init__(self):
        from src.nlp_pipeline import embed
        self._embed = embed

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return self._embed(texts, show_progress=False)


def load_all() -> None:
    global _clf, _embedder, _embeddings, _corpus_texts, _corpus_df, _etl

    onnx_cls = ONNX_DIR / "classifier"
    onnx_emb = ONNX_DIR / "embedder"
    tfidf_path = MODEL_DIR / "tfidf_pipeline.joblib"

    # Classifier: prefer the int8 ONNX DistilBERT+LoRA, else TF-IDF fallback.
    if (onnx_cls / "model_int8.onnx").exists() or (onnx_cls / "model.onnx").exists():
        try:
            _clf = WorkOrderClassifier(mode="onnx").load()
            print("[predictor] ONNX int8 DistilBERT+LoRA classifier loaded.")
        except Exception as e:
            print(f"[predictor] ONNX classifier unavailable: {e}")
    if _clf is None and tfidf_path.exists():
        try:
            _clf = WorkOrderClassifier(mode="tfidf").load()
            print("[predictor] TF-IDF classifier loaded (fallback).")
        except Exception as e:
            print(f"[predictor] TF-IDF classifier unavailable: {e}")
    if _clf is None:
        print("[predictor] No classifier artifact available.")

    # Embedder for similarity search: prefer ONNX, else sentence-transformers.
    if (onnx_emb / "model_int8.onnx").exists() or (onnx_emb / "model.onnx").exists():
        try:
            _embedder = OnnxEmbedder(str(onnx_emb))
            print("[predictor] ONNX int8 embedder loaded.")
        except Exception as e:
            print(f"[predictor] ONNX embedder unavailable: {e}")
    if _embedder is None:
        try:
            _embedder = _STEmbedder()
            print("[predictor] sentence-transformers embedder loaded (fallback).")
        except Exception as e:
            print(f"[predictor] No embedder available: {e} — similarity disabled.")

    # Embeddings index — prefer the ONNX-built index (matches the ONNX embedder),
    # fall back to the legacy index next to the other model artifacts.
    for idx_dir in (ONNX_DIR, MODEL_DIR):
        try:
            _embeddings = np.load(idx_dir / "embeddings_index.npy")
            _corpus_texts = json.loads((idx_dir / "embeddings_texts.json").read_text())
            print(f"[predictor] Embeddings index loaded from {idx_dir}: {_embeddings.shape}")
            break
        except Exception:
            continue
    if _embeddings is None:
        print("[predictor] No embeddings index — similarity search disabled.")

    # Corpus DataFrame (rows referenced by similar_cases; index aligns with the .npy)
    wo_csv = DATA_DIR / "work_orders.csv"
    if wo_csv.exists():
        _corpus_df = pd.read_csv(wo_csv)

    # ETL extractor (rule-based — no API key needed)
    _etl = ETLExtractor(mode="rule_based")


def classify(req: ClassifyRequest, top_k: int = 3) -> ClassifyResponse:
    if _clf is None:
        raise RuntimeError("Classifier not loaded.")

    result = _clf.classify(req.text)

    # Similarity search
    similar_cases = None
    if _embedder is not None and _embeddings is not None and _corpus_df is not None:
        try:
            q_emb = _embedder.encode([req.text])[0]
            hits = cosine_similarity_search(q_emb, _embeddings, top_k=top_k)
            similar_cases = []
            for idx, score in hits:
                row = _corpus_df.iloc[idx]
                similar_cases.append(SimilarCase(
                    work_order_id=str(row.get("work_order_id", idx)),
                    text=str(row.get("text", ""))[:300],
                    failure_category=str(row.get("failure_category", "")),
                    similarity_score=round(score, 4),
                ))
        except Exception as e:
            print(f"[predictor] Similarity search failed: {e}")

    # ETL extraction
    extracted = None
    if _etl is not None:
        try:
            fields = _etl.extract(req.text)
            extracted = fields.model_dump(exclude={"confidence", "extractor_used"})
        except Exception:
            pass

    return ClassifyResponse(
        category=result["category"],
        confidence=result["confidence"],
        all_scores=result["all_scores"],
        model_used=result["model_used"],
        similar_cases=similar_cases,
        extracted_fields=extracted,
    )


def is_ready() -> bool:
    return _clf is not None


def status() -> dict:
    if _clf is None:
        mode = "none"
    elif _clf.mode == "onnx":
        mode = "distilbert_lora (int8 onnx)"
    else:
        mode = _clf.mode
    return {
        "classifier_loaded": _clf is not None,
        "embeddings_loaded": _embeddings is not None,
        "model_mode": mode,
        "corpus_size": len(_corpus_df) if _corpus_df is not None else 0,
    }
