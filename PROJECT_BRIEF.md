# Project Brief — Maintenance Work Order NLP

| Priority Score | Tier | Recommended Ship Slot | Effort |
|----------------|------|----------------------|--------|
| **3.95** | **P2** | **Order #8** *(after CMAPSS · Retail Returns · RAG · HVAC · Supply Chain · Energy Demand · Industrial Failure Classification)* | 22–27 hrs across 5–6 sessions (expanded April 2026 — LoRA/QLoRA track added; expanded April 2026 again — LLM-assisted ETL standardization track added from Apr 2026 classmate method-depth pass) |

**Score breakdown** — ED 4 · DIFF 4.5 · SC 4 · DSS 5 · BV 3 · EE 3 *(DIFF nudged 4 → 4.5 after the Apr 2026 classmate method-depth pass — three flavors of LLM use now coexist in the brief: as model (PEFT), as data engineer (LLM-ETL), and indirectly as application-target via the downstream pipeline)*
**Lane:** A (Industrial — NLP angle)
**Target companies:** Any manufacturer with a CMMS (Siemens, Honeywell, GE, Boeing, every major industrial company)

**Conditions to re-rank:**
- **BV = 3** is the lowest among P2 briefs because of synthetic-data risk. Hiring managers will grade synthetic NLP data harder than synthetic tabular data — text "looks fake" more easily than rows. Prefer real public corpora when available.
- If a publicly-released CMMS / work-order dataset surfaces (e.g. Excavator CMMS dataset, oil-and-gas inspection corpus): promote BV to 4–5 and the score to ~4.10 → reconsider for P1
- If applying to a Boeing / GE Vernova / Honeywell role that emphasizes technician-facing tooling or CMMS analytics: promote tactically
- DSS = 5 (highest in portfolio, tied with HVAC + Supply Chain + Subsea) — Alvin can write authentic work orders no other candidate can

**April 2026 peer-benchmark additions (see `portfolio_pipeline.md` → *Strategic Pass — April 2026 ("Market-Signal Audit")*):**
- **Added: LoRA / QLoRA / PEFT fine-tuning track** — "I fine-tuned a small open-weight model with QLoRA for a domain-specific task" is a near-required 2026 MLE-interview talking point. DistilBERT full-fine-tune → Llama-3 8B or Mistral 7B via QLoRA as stretch. ~4 extra hours.
- This is now the **primary LLM-fine-tuning entry** in the portfolio (RAG Engineering Assistant uses GPT-4o-mini API only — no weight updates).

**April 2026 classmate method-depth pass additions (see `portfolio_pipeline.md` → *Method-Depth Pass — April 2026*):**
- **Added: LLM-assisted ETL standardization track** — A **UW MSDS classmate's** clinical-trials-style project demonstrates a story arc this brief was missing: *raw unstructured text → rule-based baseline extraction → LLM-augmented extraction → measured uplift in extraction accuracy*. That profile reported a large jump on structured-field extraction (e.g. ~63% → ~95% on protocol-style matching); we'll measure the same kind of uplift on extracted maintenance entities (equipment, failure mode, parts replaced, root cause) against ground-truth labels from the synthetic generator. ~4 extra hours.
- This adds an "applied LLM as a data-engineering tool" story to the brief — distinct from "LLM as the model" (PEFT fine-tune track) or "LLM as the application" (RAG Engineering Assistant). All three flavors of LLM use now sit in the portfolio.

---

## Problem Statement

Every industrial company generates thousands of maintenance work orders per year — free text written by technicians describing what broke, what they did, and what parts they used. This data is almost never analyzed systematically. Given a corpus of maintenance work orders, can we: (1) classify them by failure type and equipment category, (2) cluster them to surface recurring failure patterns, and (3) build a tool that helps engineers quickly find similar past failures?

---

## Why This Project for Alvin

- **Adds NLP to the portfolio:** The only text-based project. Completely different data type from all other projects — shows genuine breadth.
- **Legitimate domain authority:** Alvin has written real maintenance work orders. He knows the language, abbreviations, and failure taxonomies. His synthetic data will be realistic in ways a CS candidate's wouldn't be.
- **Immediately practical:** Every company with a CMMS (Computerized Maintenance Management System) has this exact problem. It's not academic.
- **Transformer models:** DistilBERT fine-tuning shows modern NLP skills that tech candidates claim but few actually deploy end-to-end.

---

## Dataset Strategy

**Order of preference: Public real → Public real + synthetic supplement → Synthetic only (with strong disclosure).**

### Tier 1 — Real public datasets (try first)

- **Excavator CMMS Maintenance dataset** (search Kaggle / mendeley.data) — real heavy-equipment work orders if available
- **Stack Exchange Mechanical Engineering** dump (public via archive.org) — real engineering problem text
- **OSHA inspection narratives** (public) — real industrial incident descriptions
- **NASA ASRS (Aviation Safety Reporting System)** narratives (public) — real maintenance and incident reports

If one of these works, the BV score jumps to 4–5 and this brief should be reconsidered for P1.

### Tier 2 — Real + synthetic hybrid

Use the public corpus to anchor vocabulary and label taxonomy, then generate additional synthetic work orders to balance categories. **Disclose clearly** which records are real vs. generated in the README.

### Tier 3 — Synthetic only (last resort)

Generate 2,000–5,000 realistic work orders across 6 failure categories:
- Mechanical failure (bearing wear, seal leak, shaft misalignment)
- Electrical failure (motor burnout, sensor fault, wiring issue)
- Hydraulic failure (pressure loss, valve malfunction, fluid contamination)
- Instrumentation failure (transmitter drift, thermocouple failure)
- Preventive maintenance (scheduled inspection, lubrication, filter change)
- Operator damage (impact damage, improper operation)

Each work order: equipment tag, date, technician ID, work order text (50–200 words), labor hours, parts used.

**If synthetic-only:** explicitly state in README that all data is synthetic and document Alvin's domain authorship as the credibility argument. *Why this is honest: Alvin has actually written maintenance work orders professionally and can defend the realism in an interview.*

---

## Tech Stack

| Layer | Tool |
|-------|------|
| Data generation | Python (synthetic) with domain vocabulary — generates **raw text + parallel ground-truth structured records** for ETL evaluation |
| Text preprocessing | NLTK or spaCy — tokenization, lemmatization, stopword removal |
| **LLM-assisted ETL extraction** *(new — Apr 2026 classmate method-depth pass)* | **OpenAI GPT-4o-mini API + structured-output mode (Pydantic schemas)**; rule-based regex baseline for comparison |
| Baseline NLP | TF-IDF + Logistic Regression / Random Forest |
| Advanced NLP — full fine-tune | HuggingFace Transformers — DistilBERT fine-tuned (classic full fine-tune baseline) |
| **Advanced NLP — PEFT fine-tune** *(new — peer-benchmark)* | **HuggingFace `peft` + `bitsandbytes` — LoRA on DistilBERT; QLoRA on a small decoder LM (Llama-3 8B or Mistral 7B) as stretch** |
| Clustering | BERTopic (transformer-based topic modeling) |
| Similarity search | sentence-transformers + cosine similarity |
| Evaluation | Classification report, confusion matrix, silhouette score, **before/after LoRA comparison (F1, GPU-minutes, trainable-parameters ratio)**, **regex-vs-LLM extraction F1 per entity field** |
| API | FastAPI: POST /classify → failure category + similar past orders |
| Frontend | Text input box → predicted category + 3 most similar past work orders |
| Environment | conda (environment.yml); Google Colab free tier for QLoRA GPU fine-tune (no paid GPU required); OpenAI API key required for ETL track (~$1–3 in API costs for evaluation) |

---

## NLP Pipeline Overview

```
Raw work order text
    ↓
Preprocessing (lowercase, remove punctuation, lemmatize)
    ↓
┌─────────────────┬──────────────────────────────┐
│ Baseline        │ Advanced                     │
│ TF-IDF vectors  │ DistilBERT embeddings        │
│ Logistic Reg.   │ Fine-tuned classifier        │
│ Random Forest   │                              │
└─────────────────┴──────────────────────────────┘
    ↓
Clustering (BERTopic)   →   Failure pattern discovery
    ↓
Similarity search       →   "Find similar past failures"
    ↓
FastAPI + Frontend
```

---

## Deliverables

1. `data/synthetic_generator.py` — generates realistic work order corpus with labels (raw text + parallel ground-truth structured fields)
2. `notebooks/01_eda.ipynb` — text length distribution, vocabulary analysis, class balance
3. **`notebooks/02_llm_etl_extraction.ipynb`** *(new — Apr 2026 classmate method-depth pass)* — rule-based vs. LLM vs. hybrid ETL comparison with per-field F1 and cost/latency table
4. `notebooks/03_preprocessing.ipynb` — cleaning pipeline, TF-IDF baseline
5. `notebooks/04_classification.ipynb` — TF-IDF vs. DistilBERT full-fine-tune comparison
6. **`notebooks/05_lora_finetune.ipynb`** *(new — peer-benchmark)* — LoRA fine-tune of DistilBERT; compare F1, GPU-minutes, and % of trainable parameters vs. full fine-tune
7. **`notebooks/06_qlora_finetune.ipynb`** *(new — stretch, Colab free GPU)* — QLoRA fine-tune of a small decoder LM (Llama-3 8B or Mistral 7B); classification via prompt + short-output head
8. `notebooks/07_clustering.ipynb` — BERTopic: discovered failure patterns
9. `notebooks/08_similarity.ipynb` — sentence-transformers for similar work order retrieval
10. **`src/etl_extractor.py`** *(new)* — pluggable rule-based / LLM / hybrid extraction backends with structured-output Pydantic schemas
11. `src/nlp_pipeline.py` — preprocessing + embedding functions
12. `src/classifier.py` — model loading + inference (supports both full-fine-tuned and LoRA-adapter-loaded variants)
13. `api/main.py` — FastAPI: classify work order + return similar past orders
14. `frontend/` — text input → classification result + similar cases card
15. `README.md` — GitHub-ready with live demo; **PEFT comparison table AND ETL extraction comparison table prominently featured**

---

## Project Phases

### Phase 1 — Data Generation + EDA (2–3 hrs)
- [ ] Write synthetic work order generator: vocabulary lists per failure category, sentence templates, realistic variation
- [ ] **Generator emits two parallel artifacts:** (a) raw work-order narrative text, (b) ground-truth structured fields per record — `equipment_tag`, `failure_mode`, `parts_replaced`, `root_cause`, `failure_category`. The structured fields are what we'll later try to *recover* from the raw text via ETL — the generator gives us "noisy real-world text + perfect ground truth" simultaneously. *(updated — Apr 2026 classmate method-depth pass)*
- [ ] Generate 3,000 work orders across 6 categories
- [ ] EDA: text length, vocabulary richness, class distribution, most common terms per category
- [ ] Preprocessing pipeline: lowercase, remove punctuation, lemmatize

### Phase 1.5 — LLM-Assisted ETL Standardization *(NEW — Apr 2026 classmate method-depth pass, 4 hrs)*

**Why this phase exists.** Real CMMS data arrives as messy free-text — *"replaced front bearing on pump P-104, looks like crap"*. Before we can train classifiers or do similarity search, we need *structured records*: `{equipment: "P-104", part_replaced: "front bearing", condition: "worn"}`. The 2026 standard for this is LLM-assisted extraction; the older standard is regex + lookup tables. This phase compares them on the same ground truth, with measured uplift.

- [ ] **Step 1 — Rule-based baseline extractor (1 hr).** Build a regex/keyword extractor that pulls each field from the raw text:
  - Equipment tags via regex (e.g., `r'\b[A-Z]-\d{3,4}\b'`)
  - Failure mode via keyword lists ("bearing", "seal", "valve", etc.)
  - Parts replaced via "replaced X" / "swapped Y" patterns
  - Root cause via keyword anchors ("due to", "caused by")
  - Evaluate per-field F1 against ground truth — **expected baseline: 60–70% across fields, lower on root cause (long-tail vocabulary)**
- [ ] **Step 2 — LLM-assisted extractor (2 hrs).** Use GPT-4o-mini with structured output (Pydantic schema) to extract the same fields. Prompt strategy:
  - Few-shot prompt with 3 hand-crafted exemplars per failure category (18 total)
  - Pydantic schema as JSON-mode response_format — forces structured output, no parsing failures
  - Batch processing with retries; cost ≈ $0.0001 per record × 3,000 records ≈ **$0.30 for the full evaluation run**
  - Evaluate per-field F1 against ground truth — **target uplift: 60–70% → 90%+, mirroring the ~63%→~95% structured-field uplift seen in the reference classmate profile**
- [ ] **Step 3 — Two-stage hybrid (1 hr) — the production-grade move.** Run regex first (cheap, instant); flag low-confidence extractions; route only those records to the LLM. This is the *real-world* ETL pattern — most companies can't afford LLM cost on every record but can on the hard 20%.
  - Measure cost reduction (hybrid uses LLM on ~20% of records → 80% cost savings)
  - Measure F1 vs. pure-LLM (should be within 1–2 pts — most records were easy for regex anyway)
- [ ] **Step 4 — Comparison table for README:**

  | Method | Equipment F1 | Failure-mode F1 | Parts F1 | Root-cause F1 | Avg cost / 1000 records | Latency p50 |
  |---|---|---|---|---|---|---|
  | Rule-based regex | 95% | 78% | 68% | 52% | $0 | <1 ms |
  | LLM-only (GPT-4o-mini) | 98% | 95% | 92% | 89% | $0.10 | ~800 ms |
  | Hybrid (regex → LLM-on-low-confidence) | 97% | 94% | 91% | 88% | $0.02 | ~150 ms avg |

- [ ] **Step 5 — Use the LLM-extracted structured records as input to the downstream classification phases.** Important — the rest of the pipeline operates on the *extracted* records, not the original ground truth, so end-to-end performance reflects realistic deployment (extraction errors propagate downstream, just like real life).
- [ ] Output: `notebooks/02_llm_etl_extraction.ipynb` with cost/latency/F1 comparison; `src/etl_extractor.py` with pluggable rule-based / LLM / hybrid backends

### Phase 2 — Baseline Classification (2–3 hrs)
- [ ] TF-IDF vectorization (1-gram + 2-gram)
- [ ] Logistic Regression + Random Forest classifiers
- [ ] Evaluate: classification report, confusion matrix
- [ ] Error analysis: which categories are most confused and why?

### Phase 3 — Transformer Classification: Full Fine-Tune (3–4 hrs)
- [ ] Load DistilBERT from HuggingFace
- [ ] Full fine-tune on labeled work order corpus (train/val/test split)
- [ ] Compare accuracy + F1 vs. TF-IDF baseline
- [ ] Note: DistilBERT should significantly outperform TF-IDF on subtle technical text

### Phase 3.5 — PEFT Fine-Tune Track *(NEW — peer-benchmark, 4–5 hrs)*
- [ ] **LoRA on DistilBERT** using `peft` library — target attention modules only; r=8, alpha=16
- [ ] Record: F1, total trainable parameters, trainable-% of full model, GPU-minutes to train
- [ ] Comparison table in README: Full fine-tune vs. LoRA — same F1, ~1% of the parameters updated, ~3× faster train
- [ ] *(Stretch, Colab free GPU)* **QLoRA on a small decoder LM** (Llama-3 8B or Mistral 7B) — 4-bit NF4 quantization + LoRA adapters; run on T4 or L4 in Colab; prompt-based classification with short output head
- [ ] Save LoRA adapter weights separately from base model (~few MB) — demonstrates the deployment advantage of PEFT
- [ ] Output: `notebooks/04_lora_finetune.ipynb` (+ `05_qlora_finetune.ipynb` if stretch completed)

### Phase 4 — Clustering + Similarity (2–3 hrs)
- [ ] BERTopic: discover latent failure themes without labels
- [ ] Compare BERTopic topics to known failure categories
- [ ] sentence-transformers: embed all work orders, implement cosine similarity search
- [ ] Demo: "input a new work order → find 3 most similar past cases"

### Phase 5 — API + Frontend (2–3 hrs)
- [ ] FastAPI: POST /classify → {category, confidence, similar_cases}
- [ ] Frontend: clean text area input → result card with category badge + similar cases
- [ ] Deploy Render + Vercel
- [ ] Update README

---

## Interview Talking Points

1. *"I generated the synthetic dataset using domain knowledge — I've written real maintenance work orders and know the failure taxonomy, abbreviations, and technical vocabulary. The data is realistic because I'm a domain expert."*
2. *"TF-IDF worked well for common failure modes but struggled on subtle cases — 'bearing noise' vs. 'shaft vibration' both involve rotating equipment but need different corrective actions. DistilBERT's contextual embeddings handled this correctly."*
3. *"BERTopic discovered sub-themes I hadn't explicitly labeled — for example, it separated 'seal leaks from fluid contamination' from 'seal leaks from pressure spikes', which are different root causes requiring different fixes."*
4. *"The similarity search is the most practically valuable feature — a technician describing a new problem can instantly see what fixed similar past failures, reducing diagnosis time."*
5. **\[new — LoRA / QLoRA\]** *"I compared full fine-tuning against LoRA — same F1 within a rounding error, but only about 1% of the parameters were actually updated and training was ~3× faster. For QLoRA, I ran the same experiment on a 7B-parameter decoder model on a free Colab T4 — 4-bit NF4 quantization plus LoRA adapters made it fit in memory. The lesson wasn't that LoRA is always better — it's that for domain-specific tasks with smaller datasets, PEFT gives you most of the accuracy for a fraction of the compute and makes model deployment dramatically cheaper because adapter weights are measured in megabytes, not gigabytes."*
6. **\[new — LLM ETL\]** *"Real CMMS data arrives as messy free-text. I built three extractors against the same ground truth — a regex baseline, a GPT-4o-mini extractor with a Pydantic-schema structured-output prompt, and a hybrid that runs regex first and only escalates the low-confidence records to the LLM. The pure-LLM extractor went from a regex baseline of 60-some-% on root-cause F1 to high-80s. The hybrid kept ~97% of the LLM accuracy at 20% of the API cost. That's the kind of cost-aware ETL design real production systems actually need — the LLM isn't always the answer; knowing when to call it is."*

---

## Success Criteria

- [ ] DistilBERT (full fine-tune) outperforms TF-IDF baseline on F1
- [ ] **LoRA fine-tune reaches F1 within ~1 pt of full fine-tune, with <5% of the parameters trained** *(new — peer-benchmark)*
- [ ] *(stretch)* QLoRA 7B model reaches F1 competitive with DistilBERT on Colab free tier
- [ ] **LLM-assisted ETL extractor improves average per-field F1 by ≥20 pts over regex baseline** *(new — Apr 2026 classmate method-depth pass)*
- [ ] **Hybrid extractor preserves ≥95% of pure-LLM accuracy at ≤30% of the API cost** *(new — Apr 2026 classmate method-depth pass)*
- [ ] README contains TWO comparison tables prominently:
  - PEFT comparison: Full FT vs. LoRA vs. QLoRA — F1, trainable params, GPU time
  - ETL comparison: Regex vs. LLM vs. Hybrid — per-field F1, cost per 1k records, latency
- [ ] BERTopic produces coherent, interpretable failure clusters
- [ ] Similarity search returns relevant results for test work orders
- [ ] Live FastAPI + Vercel demo
- [ ] Resume bullet: *"Built NLP pipeline for maintenance work order classification, extraction, and similarity search; designed LLM-assisted ETL extractor (GPT-4o-mini with structured-output Pydantic schemas) achieving X-pt per-field F1 uplift over regex baseline at <$0.10 per 1k records via cost-aware regex/LLM hybrid; fine-tuned DistilBERT (full + LoRA) on the resulting structured corpus while training <5% of parameters via PEFT; deployed as FastAPI with semantic search."*

---

*Brief created: April 2026 · Updated April 2026 (synthetic-data risk flagged) · Updated April 2026 (peer-benchmark pass — LoRA/QLoRA/PEFT fine-tuning track added; primary LLM-fine-tuning entry in portfolio) · Updated April 2026 (classmate method-depth pass — LLM-assisted ETL standardization track added; effort 18–22 → 22–27 hrs; priority score 3.90 → 3.95 from added DIFF coverage) · May 2026 (activated; slot confirmed #8) | Tier P2 · Ship slot #8 · NLP flagship — deepest project in portfolio (LLM ETL + PEFT + BERTopic + similarity search)*
