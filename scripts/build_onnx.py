#!/usr/bin/env python3
"""
scripts/build_onnx.py — one-time build of the int8 ONNX artifacts served in production.

WHY
    The DistilBERT+LoRA classifier and the MiniLM similarity embedder do not fit
    Render's 512 MB free tier under PyTorch (torch + both models ≈ 650 MB). Exported
    to ONNX and dynamically quantized to int8 they run under onnxruntime in
    ~250-300 MB with negligible accuracy loss — so the *real* fine-tuned model
    serves live, for free. Build offline, serve online.

RUN once, on a machine that has PyTorch (e.g. your laptop):
    pip install -r requirements-build.txt
    python scripts/build_onnx.py

OUTPUTS (committed to git; every file < 100 MB):
    models/onnx/classifier/model_int8.onnx   (+ tokenizer files)
    models/onnx/embedder/model_int8.onnx     (+ tokenizer files)
    models/onnx/embeddings_index.npy         (corpus re-embedded with the ONNX embedder)
    models/onnx/embeddings_texts.json

The script self-validates: it compares int8-ONNX predictions against the PyTorch
DistilBERT+LoRA on a labeled sample and aborts if macro accuracy drops too far.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ADAPTER_DIR = MODELS / "lora_adapter"
OUT = MODELS / "onnx"
CLS_DIR = OUT / "classifier"
EMB_DIR = OUT / "embedder"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
ACC_TOLERANCE = 0.02  # int8 must stay within 2 pts of the fp32 torch model

CATEGORY_LABELS = [
    "mechanical_failure", "electrical_failure", "hydraulic_failure",
    "instrumentation_failure", "preventive_maintenance", "operator_damage",
]


def _mean_pool(last_hidden, attention_mask):
    mask = attention_mask[..., None].astype(np.float32)
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return summed / counts


def main() -> int:
    import torch
    from transformers import AutoTokenizer, AutoModel, DistilBertForSequenceClassification
    from peft import PeftModel
    from onnxruntime.quantization import quantize_dynamic, QuantType

    def export_onnx(model, args, path, input_names, output_names, dynamic_axes):
        # Force the LEGACY TorchScript exporter (dynamo=False): it writes a single,
        # self-contained .onnx that quantize_dynamic can consume. The newer dynamo
        # exporter (torch >= 2.6, which becomes the default once onnxscript is
        # installed) writes an external-data pair (model.onnx + model.onnx.data,
        # ~256 MB) that both breaks quantization and blows past GitHub's 100 MB limit.
        kw = dict(input_names=input_names, output_names=output_names,
                  dynamic_axes=dynamic_axes, opset_version=14)
        try:
            torch.onnx.export(model, args, str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, args, str(path), **kw)  # torch < 2.5: no dynamo kwarg

    def finalize_int8(model_dir, fp32_path, int8_path):
        # Safety net: if the export still wrote external data (newer dynamo exporter),
        # consolidate to a single self-contained file so quantize_dynamic can read it
        # and nothing >100 MB survives the build.
        data_blob = model_dir / (fp32_path.name + ".data")
        if data_blob.exists():
            import onnx
            onnx.save_model(onnx.load(str(fp32_path)), str(fp32_path), save_as_external_data=False)
            data_blob.unlink()
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
        assert int8_path.exists() and int8_path.stat().st_size > 0, \
            f"int8 quantization produced nothing at {int8_path}"
        # Keep ONLY the int8 file in git — remove the fp32 graph and any external-data blob.
        for pat in ("model.onnx", "model.onnx.data", "model.onnx_data"):
            for f in model_dir.glob(pat):
                f.unlink()

    OUT.mkdir(parents=True, exist_ok=True)
    CLS_DIR.mkdir(parents=True, exist_ok=True)
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Merge LoRA into the stock base ────────────────────────────────────
    cfg = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    base_id = cfg.get("base_model_name_or_path", "distilbert-base-uncased")
    print(f"[build] base = {base_id}  (+ LoRA adapter, incl. trained head via modules_to_save)")
    cls_tok = AutoTokenizer.from_pretrained(base_id)
    base = DistilBertForSequenceClassification.from_pretrained(base_id, num_labels=len(CATEGORY_LABELS))
    merged = PeftModel.from_pretrained(base, str(ADAPTER_DIR)).merge_and_unload().eval()

    # ── 2. Export classifier to ONNX, then int8 ──────────────────────────────
    merged.config.return_dict = False
    dummy = cls_tok("bearing failure on pump", return_tensors="pt", truncation=True, max_length=256)
    cls_fp32 = CLS_DIR / "model.onnx"
    cls_int8 = CLS_DIR / "model_int8.onnx"
    export_onnx(merged, (dummy["input_ids"], dummy["attention_mask"]), cls_fp32,
                ["input_ids", "attention_mask"], ["logits"],
                {"input_ids": {0: "b", 1: "s"}, "attention_mask": {0: "b", 1: "s"}, "logits": {0: "b"}})
    finalize_int8(CLS_DIR, cls_fp32, cls_int8)
    cls_tok.save_pretrained(str(CLS_DIR))
    print(f"[build] classifier int8 → {cls_int8}  ({cls_int8.stat().st_size/1e6:.1f} MB)")

    # ── 3. Export MiniLM embedder to ONNX, then int8 ─────────────────────────
    emb_tok = AutoTokenizer.from_pretrained(EMBED_MODEL_ID)
    emb_model = AutoModel.from_pretrained(EMBED_MODEL_ID).eval()
    emb_model.config.return_dict = False
    d = emb_tok("bearing failure on pump", return_tensors="pt", truncation=True, max_length=256)
    emb_inputs = (d["input_ids"], d["attention_mask"], d["token_type_ids"])
    emb_fp32 = EMB_DIR / "model.onnx"
    emb_int8 = EMB_DIR / "model_int8.onnx"
    export_onnx(emb_model, emb_inputs, emb_fp32,
                ["input_ids", "attention_mask", "token_type_ids"], ["last_hidden_state"],
                {k: {0: "b", 1: "s"} for k in ["input_ids", "attention_mask", "token_type_ids", "last_hidden_state"]})
    finalize_int8(EMB_DIR, emb_fp32, emb_int8)
    emb_tok.save_pretrained(str(EMB_DIR))
    print(f"[build] embedder int8 → {emb_int8}  ({emb_int8.stat().st_size/1e6:.1f} MB)")

    # ── 4. Re-embed the corpus with the ONNX embedder (index ⋈ work_orders.csv) ─
    import onnxruntime as ort
    import pandas as pd
    wo_csv = ROOT / "data" / "work_orders.csv"
    if not wo_csv.exists():
        print("[build] work_orders.csv missing — regenerating (seeded)…")
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "data" / "synthetic_generator.py")], check=True)
    df = pd.read_csv(wo_csv)
    texts = df["text"].fillna("").astype(str).tolist()

    emb_sess = ort.InferenceSession(str(emb_int8), providers=["CPUExecutionProvider"])
    def embed_batch(batch):
        enc = emb_tok(batch, return_tensors="np", padding=True, truncation=True, max_length=256)
        feeds = {i.name: enc[i.name].astype(np.int64) for i in emb_sess.get_inputs()}
        last_hidden = emb_sess.run(None, feeds)[0]
        return _mean_pool(last_hidden, enc["attention_mask"])
    vecs = []
    for i in range(0, len(texts), 64):
        vecs.append(embed_batch(texts[i:i + 64]))
        print(f"\r[build] embedding corpus {min(i+64,len(texts))}/{len(texts)}", end="", flush=True)
    print()
    index = np.vstack(vecs).astype(np.float32)
    np.save(OUT / "embeddings_index.npy", index)
    (OUT / "embeddings_texts.json").write_text(json.dumps(texts))
    print(f"[build] embeddings index → {index.shape}")

    # ── 5. Validate int8 vs torch on a labeled sample ────────────────────────
    cls_sess = ort.InferenceSession(str(cls_int8), providers=["CPUExecutionProvider"])
    sample = df.sample(min(500, len(df)), random_state=0)
    y_true = sample["failure_category"].tolist()
    onnx_pred, torch_pred = [], []
    for i in range(0, len(sample), 32):
        chunk = sample["text"].fillna("").astype(str).tolist()[i:i + 32]
        enc = cls_tok(chunk, return_tensors="np", padding=True, truncation=True, max_length=256)
        feeds = {inp.name: enc[inp.name].astype(np.int64) for inp in cls_sess.get_inputs()}
        logits = cls_sess.run(None, feeds)[0]
        onnx_pred += [CATEGORY_LABELS[j] for j in logits.argmax(1)]
        with torch.no_grad():
            tlog = merged(input_ids=torch.tensor(enc["input_ids"]),
                          attention_mask=torch.tensor(enc["attention_mask"]))[0]
        torch_pred += [CATEGORY_LABELS[j] for j in tlog.argmax(1).tolist()]
    onnx_acc = np.mean([p == t for p, t in zip(onnx_pred, y_true)])
    torch_acc = np.mean([p == t for p, t in zip(torch_pred, y_true)])
    agree = np.mean([o == t for o, t in zip(onnx_pred, torch_pred)])
    print(f"\n[validate] torch fp32 acc={torch_acc:.3f} | onnx int8 acc={onnx_acc:.3f} "
          f"| int8↔torch agreement={agree:.3f}  (n={len(sample)})")
    if onnx_acc < torch_acc - ACC_TOLERANCE:
        print(f"[validate] ✗ int8 dropped >{ACC_TOLERANCE:.0%} vs fp32 — NOT shipping. "
              f"Consider per-channel/QDQ quantization.")
        return 1
    print("[validate] ✓ int8 accuracy holds. Artifacts ready to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
