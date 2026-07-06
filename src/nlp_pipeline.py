"""
src/nlp_pipeline.py — Text preprocessing and embedding for maintenance work orders.

Functions:
    preprocess(text) → cleaned text
    tfidf_vectorize(texts, max_features) → sparse matrix + vectorizer
    embed(texts, model_name) → numpy array of dense embeddings (sentence-transformers)
    load_embeddings(path) / save_embeddings(embeddings, texts, path) — persistence

Model choices:
    - Embeddings: 'all-MiniLM-L6-v2' (fast, 384-dim, good for short text)
    - Alternative: 'all-mpnet-base-v2' (slower, 768-dim, higher quality)
"""

from __future__ import annotations

import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# ─── Preprocessing ────────────────────────────────────────────────────────────

# Industry abbreviations to expand before cleaning
ABBREV_MAP = {
    r'\bpt\b': 'pressure transmitter',
    r'\btt\b': 'temperature transmitter',
    r'\bft\b': 'flow transmitter',
    r'\blt\b': 'level transmitter',
    r'\bvfd\b': 'variable frequency drive',
    r'\bpm\b': 'preventive maintenance',
    r'\bwo\b': 'work order',
    r'\bprv\b': 'pressure relief valve',
    r'\bhx\b': 'heat exchanger',
    r'\bfan\b': 'fan unit',
    r'\bdcs\b': 'distributed control system',
    r'\bplc\b': 'programmable logic controller',
    r'\bscada\b': 'supervisory control and data acquisition',
    r'\bir\b': 'infrared',
}

# Common stop words plus domain-specific noise words
STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "been", "be",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "no", "nor",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "we", "our", "he", "she", "him", "her",
    "also", "then", "than", "so", "as", "if", "when", "where",
    # Domain noise
    "equipment", "unit", "noted", "found", "confirmed", "reported",
    "operator", "technician", "performed", "returned", "service",
    "work", "order", "maintenance", "wo",
}


def preprocess(text: str, expand_abbrevs: bool = True) -> str:
    """
    Clean and normalize work order text.
    Steps: lowercase → expand abbreviations → remove punctuation → strip stop words → normalize spaces.
    """
    t = text.lower().strip()
    if expand_abbrevs:
        for pattern, replacement in ABBREV_MAP.items():
            t = re.sub(pattern, replacement, t)
    t = t.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
    tokens = [tok for tok in t.split() if tok not in STOP_WORDS and len(tok) > 1]
    return ' '.join(tokens)


def preprocess_series(texts: pd.Series, expand_abbrevs: bool = True) -> pd.Series:
    return texts.fillna('').apply(lambda t: preprocess(t, expand_abbrevs))


# ─── TF-IDF ───────────────────────────────────────────────────────────────────

def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )


def fit_tfidf(texts: list[str], max_features: int = 5000):
    """Fit TF-IDF and return (vectorizer, sparse matrix)."""
    vec = build_tfidf_vectorizer(max_features)
    X = vec.fit_transform(texts)
    return vec, X


# ─── Sentence-transformer embeddings ─────────────────────────────────────────

def embed(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute dense sentence embeddings.

    Returns shape (n_texts, embedding_dim).
    Model 'all-MiniLM-L6-v2': 384-dim, fast, good for short domain text.
    """
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )
    return embeddings


class OnnxEmbedder:
    """Mean-pooled MiniLM embeddings via onnxruntime (int8) — no torch, no
    sentence-transformers. Reproduces the mean pooling used to build the corpus
    index; L2 normalization is applied in cosine_similarity_search, so ranking is
    unaffected. This is what the API uses in production (fits Render's free tier)."""

    def __init__(self, onnx_dir: str = "models/onnx/embedder"):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        d = Path(onnx_dir)
        model_path = d / "model_int8.onnx"
        if not model_path.exists():
            model_path = d / "model.onnx"
        self._tok = AutoTokenizer.from_pretrained(str(d))
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            str(model_path), sess_options=so, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), batch_size):
            enc = self._tok(texts[i:i + batch_size], return_tensors="np",
                            padding=True, truncation=True, max_length=256)
            feeds = {inp.name: enc[inp.name].astype(np.int64) for inp in self._sess.get_inputs()}
            last_hidden = self._sess.run(None, feeds)[0]
            mask = enc["attention_mask"][..., None].astype(np.float32)
            summed = (last_hidden * mask).sum(axis=1)
            counts = np.clip(mask.sum(axis=1), 1e-9, None)
            out.append(summed / counts)
        return np.vstack(out).astype(np.float32)


def save_embeddings(embeddings: np.ndarray, texts: list[str], out_dir: str = "models") -> None:
    path = Path(out_dir)
    path.mkdir(exist_ok=True)
    np.save(path / "embeddings_index.npy", embeddings)
    (path / "embeddings_texts.json").write_text(json.dumps(texts))
    print(f"[nlp_pipeline] Saved embeddings {embeddings.shape} → {path}")


def load_embeddings(model_dir: str = "models") -> tuple[np.ndarray, list[str]]:
    path = Path(model_dir)
    embeddings = np.load(path / "embeddings_index.npy")
    texts = json.loads((path / "embeddings_texts.json").read_text())
    return embeddings, texts


# ─── Cosine similarity search ─────────────────────────────────────────────────

def cosine_similarity_search(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    top_k: int = 3,
) -> list[tuple[int, float]]:
    """
    Find top_k most similar embeddings by cosine similarity.

    Returns list of (index, score) tuples, sorted descending by score.
    """
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    corpus_norm = corpus_embeddings / (np.linalg.norm(corpus_embeddings, axis=1, keepdims=True) + 1e-10)
    scores = corpus_norm @ query_norm
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i])) for i in top_indices]


# ─── Label encoding ───────────────────────────────────────────────────────────

CATEGORY_LABELS = [
    "mechanical_failure",
    "electrical_failure",
    "hydraulic_failure",
    "instrumentation_failure",
    "preventive_maintenance",
    "operator_damage",
]


def encode_labels(series: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    le.classes_ = np.array(CATEGORY_LABELS)
    y = le.transform(series)
    return y, le
