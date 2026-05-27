"""
src/classifier.py — Model loading and inference for work order classification.

Supports three model variants:
    1. TF-IDF + Logistic Regression (baseline, fast, no GPU required)
    2. DistilBERT full fine-tune (HuggingFace Transformers)
    3. DistilBERT + LoRA adapters (PEFT, ~1% of parameters trained)

Usage:
    clf = WorkOrderClassifier(mode="distilbert_lora")
    result = clf.classify("Replaced mechanical seal on pump P-104 after bearing failure.")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

CATEGORY_LABELS = [
    "mechanical_failure",
    "electrical_failure",
    "hydraulic_failure",
    "instrumentation_failure",
    "preventive_maintenance",
    "operator_damage",
]

MODEL_DIR = Path("models")


class WorkOrderClassifier:
    """
    Unified classifier interface across TF-IDF, DistilBERT, and LoRA variants.

    Args:
        mode: "tfidf" | "distilbert" | "distilbert_lora"
        model_dir: directory containing saved model artifacts
    """

    def __init__(self, mode: str = "distilbert_lora", model_dir: str = str(MODEL_DIR)):
        if mode not in ("tfidf", "distilbert", "distilbert_lora"):
            raise ValueError("mode must be 'tfidf', 'distilbert', or 'distilbert_lora'")
        self.mode = mode
        self.model_dir = Path(model_dir)
        self._model = None
        self._tokenizer = None
        self._tfidf_pipeline = None
        self._loaded = False

    def load(self) -> "WorkOrderClassifier":
        if self.mode == "tfidf":
            self._tfidf_pipeline = joblib.load(self.model_dir / "tfidf_pipeline.joblib")
        elif self.mode == "distilbert":
            self._load_hf_model(self.model_dir / "distilbert_finetuned")
        elif self.mode == "distilbert_lora":
            self._load_lora_model(self.model_dir / "lora_adapter")
        self._loaded = True
        return self

    def _load_hf_model(self, path: Path) -> None:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        import torch
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(path))
        self._model = DistilBertForSequenceClassification.from_pretrained(str(path))
        self._model.eval()
        self._device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._model.to(self._device)

    def _load_lora_model(self, adapter_path: Path) -> None:
        from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
        from peft import PeftModel
        import torch
        base_path = self.model_dir / "distilbert_finetuned"
        self._tokenizer = DistilBertTokenizerFast.from_pretrained(str(base_path))
        base_model = DistilBertForSequenceClassification.from_pretrained(str(base_path))
        self._model = PeftModel.from_pretrained(base_model, str(adapter_path))
        self._model.eval()
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)

    def classify(self, text: str) -> dict:
        """
        Classify a single work order text.

        Returns:
            {
                "category": str,
                "confidence": float,
                "all_scores": {category: score, ...},
                "model_used": str,
            }
        """
        if not self._loaded:
            self.load()

        if self.mode == "tfidf":
            return self._classify_tfidf(text)
        return self._classify_hf(text)

    def _classify_tfidf(self, text: str) -> dict:
        proba = self._tfidf_pipeline.predict_proba([text])[0]
        best_idx = int(np.argmax(proba))
        return {
            "category":   CATEGORY_LABELS[best_idx],
            "confidence": round(float(proba[best_idx]), 4),
            "all_scores": {CATEGORY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
            "model_used": "tfidf_lr",
        }

    def _classify_hf(self, text: str) -> dict:
        import torch
        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256, padding=True
        )
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self._model(**inputs).logits
        proba = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        best_idx = int(np.argmax(proba))
        model_tag = "distilbert_lora" if self.mode == "distilbert_lora" else "distilbert_full"
        return {
            "category":   CATEGORY_LABELS[best_idx],
            "confidence": round(float(proba[best_idx]), 4),
            "all_scores": {CATEGORY_LABELS[i]: round(float(p), 4) for i, p in enumerate(proba)},
            "model_used": model_tag,
        }

    @property
    def is_loaded(self) -> bool:
        return self._loaded
