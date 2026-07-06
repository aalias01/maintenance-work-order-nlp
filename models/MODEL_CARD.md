# Model card

## Classifier

The served classifier is DistilBERT fine-tuned with LoRA adapters on a 3,000-record synthetic maintenance work-order corpus. On the held-out 600-record test set, stratified with seed 42, it reaches 94.4% macro F1 while training 743K parameters, 1.1% of the model.

Reference results from the README:

| Approach | Macro F1 | Parameters trained | Artifact |
|---|---:|---:|---:|
| TF-IDF and logistic regression | 93.5% | 27K | 0.4 MB |
| DistilBERT full fine-tune | 95.9% | 67.0M | 256 MB |
| DistilBERT with LoRA | 94.4% | 743K | 2.8 MB |

The production artifact is the LoRA-merged classifier exported to int8 ONNX. The API serves it with onnxruntime and tokenizer files only. Torch is not installed in the production runtime. If the ONNX classifier is missing, the API can fall back to the TF-IDF pipeline.

## Extraction

The repo measures three extraction backends on a 100-record sample with `random_state=42`.

| Method | Equipment tag | Failure mode | Parts | Root cause | Category | LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| Rule-based regex | 82% | 13% | 40% | 42% | 50% | 0% |
| LLM-only, GPT-4o-mini | 99% | 70% | 80% | 77% | 72% | 100% |
| Hybrid, regex then LLM below 0.7 confidence | 96% | 23% | 58% | 58% | 59% | 44% |

The live API uses the rule-based extractor at serve time. The 70% failure-mode result is a notebook result, not the endpoint. The hybrid result is the main negative finding: it cut LLM calls by 56% but kept far less than 56% of the uplift because regex confidence was miscalibrated on the noisy records.

## Retrieval

Semantic retrieval uses a MiniLM sentence-embedding index and cosine similarity to return the top 3 closest past cases. Production does not ship the ignored training CSV, so `models/corpus_meta.json` carries the aligned `work_order_id`, `failure_category`, and capped text for each embedding row. The build script asserts that metadata length matches the embeddings matrix and spot-checks the first, middle, and last rows against `embeddings_texts.json`.

## Data

The corpus has 3,000 synthetic work orders. The generator uses a calibrated noise layer: character typos, technician shorthand, vague notes, confusable symptoms, terse one-liners, and about 2% label noise. The taxonomy, vocabulary, and abbreviations come from 12 years of writing real work orders across Rheem, Centurion, Baker Hughes, and Daikin.

## Limitations

The corpus is synthetic and English-only. It has not been trained against a real CMMS export with date quirks, pasted boilerplate, multilingual notes, or site-specific shorthand. Serve-time extraction is rule-based, so narrative fields remain the weakest part of the live endpoint. The QLoRA notebook is a scaffolded stretch path and needs a Colab GPU.
