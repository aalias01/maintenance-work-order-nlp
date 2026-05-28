from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.classifier import WorkOrderClassifier
from src.nlp_pipeline import embed, cosine_similarity_search
from src.etl_extractor import ETLExtractor
from api.schemas import ClassifyRequest, ClassifyResponse, SimilarCase

MODEL_DIR = Path("models")
DATA_DIR  = Path("data")

_clf: Optional[WorkOrderClassifier] = None
_embeddings: Optional[np.ndarray] = None
_corpus_texts: Optional[list[str]] = None
_corpus_df: Optional[pd.DataFrame] = None
_etl: Optional[ETLExtractor] = None
_emb_model_name = "all-MiniLM-L6-v2"


def load_all() -> None:
    global _clf, _embeddings, _corpus_texts, _corpus_df, _etl

    # Classifier
    lora_dir = MODEL_DIR / "lora_adapter"
    distilbert_dir = MODEL_DIR / "distilbert_finetuned"
    tfidf_path = MODEL_DIR / "tfidf_pipeline.joblib"
    if lora_dir.exists() and distilbert_dir.exists():
        try:
            _clf = WorkOrderClassifier(mode="distilbert_lora").load()
            print("[predictor] LoRA classifier loaded.")
        except Exception as e:
            print(f"[predictor] LoRA classifier unavailable: {e}")
    if _clf is None and distilbert_dir.exists():
        try:
            _clf = WorkOrderClassifier(mode="distilbert").load()
            print("[predictor] DistilBERT classifier loaded.")
        except Exception as e:
            print(f"[predictor] DistilBERT classifier unavailable: {e}")
    if _clf is None and tfidf_path.exists():
        try:
            _clf = WorkOrderClassifier(mode="tfidf").load()
            print("[predictor] TF-IDF classifier loaded.")
        except Exception as e:
            print(f"[predictor] TF-IDF classifier unavailable: {e}")
    if _clf is None:
        print("[predictor] No classifier artifact available.")

    # Embeddings index
    try:
        _embeddings = np.load(MODEL_DIR / "embeddings_index.npy")
        _corpus_texts = json.loads((MODEL_DIR / "embeddings_texts.json").read_text())
        print(f"[predictor] Embeddings loaded: {_embeddings.shape}")
    except Exception as e:
        print(f"[predictor] No embeddings: {e} — similarity search disabled.")

    # Corpus DataFrame
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
    if _embeddings is not None and _corpus_df is not None:
        try:
            q_emb = embed([req.text], show_progress=False)[0]
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
    return {
        "classifier_loaded": _clf is not None,
        "embeddings_loaded": _embeddings is not None,
        "model_mode": _clf.mode if _clf else "none",
        "corpus_size": len(_corpus_df) if _corpus_df is not None else 0,
    }
