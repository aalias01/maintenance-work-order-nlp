# Portfolio Readiness Guide

This project is a strong portfolio scaffold, but it should not be presented as finished until the notebooks have been run, the result tables have real metrics, and the API/frontend have been deployed.

## Current State

Completed:

- End-to-end repository scaffold for synthetic maintenance work-order NLP.
- Synthetic data generator for 3,000 balanced work orders across 6 failure categories.
- Rule-based, LLM, and hybrid ETL extractor interfaces.
- Classifier wrapper for TF-IDF, DistilBERT, and DistilBERT + LoRA modes.
- FastAPI service with `/health` and `/classify`.
- Vanilla frontend for classification, extracted fields, and similar cases.
- Render deployment config and frontend-ready static files.
- Local synthetic corpus generated during readiness review: `data/work_orders.csv` and `data/work_orders_ground_truth.json`.
- Local baseline model saved during readiness review: `models/tfidf_pipeline.joblib`.
- Local semantic retrieval index saved during readiness review: `models/embeddings_index.npy` and `models/embeddings_texts.json`.

Still required before portfolio publishing:

- Run notebooks and replace placeholder README metrics with measured values.
- Add final screenshots or figures from EDA/model evaluation.
- Deploy API and frontend, then replace placeholder live links.
- Commit and push the final tracked artifacts.

## Portfolio Readiness Checklist

### 1. Create the Environment

```bash
conda env create -f environment.yml
conda activate maintenance-nlp
python -m ipykernel install --user --name maintenance-nlp --display-name "maintenance-nlp"
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate maintenance-nlp
```

### 2. Configure Secrets

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=your_real_key_here
```

Do not commit `.env`. It is already gitignored.

### 3. Generate the Dataset

```bash
python data/synthetic_generator.py
```

Expected local outputs:

- `data/work_orders.csv`
- `data/work_orders_ground_truth.json`

These files are gitignored so the repo stays lightweight. The README should state that users can regenerate them.

### 4. Run the Notebooks in Order

Run these locally in Jupyter using the `maintenance-nlp` kernel:

```text
notebooks/01_eda.ipynb
notebooks/02_llm_etl_extraction.ipynb
notebooks/03_preprocessing.ipynb
notebooks/04_classification.ipynb
notebooks/05_lora_finetune.ipynb
notebooks/07_clustering.ipynb
notebooks/08_similarity.ipynb
```

Notebook `06_qlora_finetune.ipynb` is a stretch notebook for Colab/CUDA. Skip it for the first portfolio release unless you have time to run it cleanly.

Minimum ship standard:

- Notebook 02 produces ETL comparison metrics.
- Notebook 03 saves `models/tfidf_pipeline.joblib`.
- Notebook 08 saves `models/embeddings_index.npy` and `models/embeddings_texts.json`.

Better ship standard:

- Notebook 04 saves `models/distilbert_finetuned/`.
- Notebook 05 saves `models/lora_adapter/`.
- README has measured full fine-tune vs. LoRA comparison values.

### 5. Update README Metrics

Replace every `—` placeholder in the README tables with measured values:

- ETL Extraction Results:
  - Equipment F1
  - Failure-mode F1
  - Parts F1
  - Root-cause F1
  - Cost per 1K records
  - Latency p50

- PEFT Comparison:
  - TF-IDF F1
  - DistilBERT full fine-tune F1
  - LoRA F1
  - Trainable parameters
  - Percent of model trained
  - GPU-minutes or CPU-minutes

Also update:

- Live demo URL
- API docs URL
- Any claims that say "same F1" or "97% of LLM accuracy" so they match your measured results.

### 6. Smoke-Test the API

Start the API:

```bash
uvicorn api.main:app --reload
```

Check health:

```bash
curl http://127.0.0.1:8000/health
```

Test classification:

```bash
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text":"Responded to high vibration alarm on P-104. Found bearing wear after inspection. Root cause: inadequate lubrication. Replaced mechanical seal and bearing set. Returned to service."}'
```

Expected:

- `status` is `ok` once a classifier is available.
- `classifier_loaded` is `true`.
- `embeddings_loaded` is `true` after notebook 08.
- `/classify` returns a category, confidence, extracted fields, and similar cases.

### 7. Smoke-Test the Frontend

With the API running locally, open:

```text
frontend/index.html
```

Use the example buttons and confirm:

- API status changes from offline to ready.
- Classification card renders.
- Extracted fields render.
- Similar past cases render after embeddings are built.
- No user/API text is rendered as raw HTML.

### 8. Prepare Deployment

Render API:

1. Connect the GitHub repo to Render.
2. Use the existing `render.yaml`.
3. Add `OPENAI_API_KEY` in the Render dashboard if you want LLM extraction available in deployment.
4. Confirm `/health` and `/docs` work after deploy.

Frontend:

1. Deploy `frontend/` as a static project on Vercel.
2. Update `frontend/app.js`:

```js
const API_BASE = 'https://your-render-service.onrender.com';
```

3. Update API CORS in `api/main.py` with the Vercel URL:

```python
"https://your-project.vercel.app",
```

4. Redeploy the API after the CORS change.

### 9. Final Local Verification

Run:

```bash
git status --short
python -m py_compile data/synthetic_generator.py src/etl_extractor.py src/nlp_pipeline.py src/classifier.py api/main.py api/predictor.py api/schemas.py
```

If `py_compile` fails on macOS because Python tries to write bytecode under `~/Library/Caches`, use:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/pycache python -m py_compile data/synthetic_generator.py src/etl_extractor.py src/nlp_pipeline.py src/classifier.py api/main.py api/predictor.py api/schemas.py
```

### 10. Commit and Push

Review changes:

```bash
git status --short
git diff
```

Stage only intended tracked files:

```bash
git add README.md docs/PORTFOLIO_READINESS.md frontend/app.js frontend/style.css
```

If you decide to commit model artifacts, stage only the small portfolio artifacts listed in `.gitignore` exceptions:

```bash
git add models/tfidf_pipeline.joblib models/embeddings_index.npy models/embeddings_texts.json
git add models/distilbert_finetuned models/lora_adapter
```

Commit:

```bash
git commit -m "Prepare maintenance NLP project for portfolio completion"
```

Push:

```bash
git push origin main
```

## Suggested Portfolio Release Standard

For the first public release, prioritize a complete working demo over the full stretch scope:

1. Ship TF-IDF classifier plus embeddings search first.
2. Include measured ETL results from regex and a small LLM sample.
3. Add DistilBERT + LoRA metrics once training is clean.
4. Treat QLoRA as a bonus section, not a blocker.

That gives recruiters a runnable app immediately while still preserving the deeper LLM/PEFT story.
