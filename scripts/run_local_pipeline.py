"""One-command reproduction of all model metrics (notebooks 03-05, 07-08).

Runs on a laptop (Apple Silicon MPS or CPU):
    python scripts/run_local_pipeline.py            # everything (~15-40 min)
    python scripts/run_local_pipeline.py --steps tfidf,distilbert,lora
    python scripts/run_local_pipeline.py --fast     # 2 epochs instead of 4

Outputs:
    models/      tfidf_pipeline.joblib, distilbert_finetuned/, lora_adapter/
    figures/     confusion matrices, BERTopic crosstab heatmap
    docs_local/run_results.json   all measured metrics (private, gitignored)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import numpy as np
import pandas as pd

from src.nlp_pipeline import preprocess_series, build_tfidf_vectorizer, CATEGORY_LABELS

RESULTS_PATH = REPO / 'docs_local' / 'run_results.json'
SEED = 42


def load_results() -> dict:
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results: dict) -> None:
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f'[saved] {RESULTS_PATH.relative_to(REPO)}')


def save_confusion_matrix(y_true, y_pred, labels, title, outfile):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                xticklabels=[c.replace('_', ' ')[:12] for c in labels],
                yticklabels=[c.replace('_', ' ')[:12] for c in labels])
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()
    plt.savefig(REPO / 'figures' / outfile, dpi=150)
    plt.close()
    print(f'[saved] figures/{outfile}')


# ---------------------------------------------------------------- TF-IDF (03)
def step_tfidf(df, results):
    import joblib
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    print('\n=== [03] TF-IDF + Logistic Regression baseline ===')
    clean = preprocess_series(df.text)
    Xtr, Xte, ytr, yte = train_test_split(
        clean, df.failure_category, test_size=0.2,
        stratify=df.failure_category, random_state=SEED)

    t0 = time.time()
    pipe = Pipeline([
        ('tfidf', build_tfidf_vectorizer(max_features=5000)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000,
                                   random_state=SEED)),
    ])
    pipe.fit(Xtr, ytr)
    minutes = (time.time() - t0) / 60
    y_pred = pipe.predict(Xte)
    f1 = f1_score(yte, y_pred, average='macro')
    print(classification_report(yte, y_pred))

    joblib.dump(pipe, REPO / 'models' / 'tfidf_pipeline.joblib')
    save_confusion_matrix(yte, y_pred, CATEGORY_LABELS,
                          'Confusion Matrix - TF-IDF + LR baseline',
                          'confusion_matrix_tfidf.png')
    n_params = pipe.named_steps['clf'].coef_.size + pipe.named_steps['clf'].intercept_.size
    results['tfidf'] = {'f1_macro': round(f1, 4),
                        'train_minutes': round(minutes, 2),
                        'trainable_params': int(n_params)}
    save_results(results)


# -------------------------------------------------- transformer helpers (04/05)
def transformer_data(df):
    from datasets import Dataset
    from sklearn.model_selection import train_test_split
    from transformers import DistilBertTokenizerFast

    label2id = {c: i for i, c in enumerate(CATEGORY_LABELS)}
    df = df.copy()
    df['label'] = df.failure_category.map(label2id)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df.label,
                                         random_state=SEED)
    train_df, val_df = train_test_split(train_df, test_size=0.1,
                                        stratify=train_df.label, random_state=SEED)
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], truncation=True, max_length=256,
                         padding='max_length')

    def mk(d):
        return Dataset.from_pandas(
            d[['text', 'label']].reset_index(drop=True)).map(tokenize, batched=True)

    return tokenizer, label2id, train_df, val_df, test_df, mk(train_df), mk(val_df), mk(test_df)


def train_transformer(model, tokenizer, train_ds, val_ds, test_ds, test_df,
                      outdir, epochs, lr, tag):
    from sklearn.metrics import classification_report, f1_score
    from transformers import Trainer, TrainingArguments

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {'f1': f1_score(labels, preds, average='macro')}

    args = TrainingArguments(
        output_dir=f'/tmp/{tag}', num_train_epochs=epochs,
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        eval_strategy='epoch', save_strategy='no',
        learning_rate=lr, warmup_ratio=0.1, weight_decay=0.01,
        logging_steps=50, report_to='none', seed=SEED,
    )
    t0 = time.time()
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=val_ds, compute_metrics=compute_metrics)
    trainer.train()
    minutes = (time.time() - t0) / 60

    preds_out = trainer.predict(test_ds)
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    f1 = f1_score(test_df.label, y_pred, average='macro')
    print(classification_report(test_df.label, y_pred,
                                target_names=CATEGORY_LABELS))
    return trainer, y_pred, f1, minutes


# ------------------------------------------------------- DistilBERT full (04)
def step_distilbert(df, results, epochs):
    from transformers import DistilBertForSequenceClassification

    print('\n=== [04] DistilBERT full fine-tune ===')
    tokenizer, label2id, *_, test_df, train_ds, val_ds, test_ds = _unpack(df)
    id2label = {i: c for c, i in label2id.items()}
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(CATEGORY_LABELS),
        id2label=id2label, label2id=label2id)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainer, y_pred, f1, minutes = train_transformer(
        model, tokenizer, train_ds, val_ds, test_ds, test_df,
        'models/distilbert_finetuned', epochs, 2e-5, 'db_full')

    trainer.save_model(str(REPO / 'models' / 'distilbert_finetuned'))
    tokenizer.save_pretrained(str(REPO / 'models' / 'distilbert_finetuned'))
    save_confusion_matrix(
        test_df.label.map({i: c for c, i in label2id.items()}),
        pd.Series(y_pred).map({i: c for c, i in label2id.items()}),
        CATEGORY_LABELS, 'Confusion Matrix - DistilBERT full fine-tune',
        'confusion_matrix_distilbert.png')
    results['distilbert_full'] = {
        'f1_macro': round(f1, 4), 'train_minutes': round(minutes, 1),
        'trainable_params': int(n_params), 'pct_of_model': 100.0,
        'epochs': epochs}
    save_results(results)


# ------------------------------------------------------------------ LoRA (05)
def step_lora(df, results, epochs):
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import DistilBertForSequenceClassification

    print('\n=== [05] LoRA fine-tune ===')
    tokenizer, label2id, *_, test_df, train_ds, val_ds, test_ds = _unpack(df)
    id2label = {i: c for c, i in label2id.items()}
    base = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=len(CATEGORY_LABELS),
        id2label=id2label, label2id=label2id)
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16,
                        lora_dropout=0.1, target_modules=['q_lin', 'v_lin'])
    model = get_peft_model(base, config)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())

    trainer, y_pred, f1, minutes = train_transformer(
        model, tokenizer, train_ds, val_ds, test_ds, test_df,
        'models/lora_adapter', epochs, 3e-4, 'db_lora')

    model.save_pretrained(str(REPO / 'models' / 'lora_adapter'))
    results['lora'] = {
        'f1_macro': round(f1, 4), 'train_minutes': round(minutes, 1),
        'trainable_params': int(n_trainable),
        'pct_of_model': round(100 * n_trainable / n_total, 2),
        'epochs': epochs}

    def dir_size_mb(p):
        return round(sum(f.stat().st_size for f in Path(p).rglob('*')
                         if f.is_file()) / 1024**2, 1)

    results['artifact_sizes_mb'] = {
        'distilbert_finetuned': dir_size_mb(REPO / 'models' / 'distilbert_finetuned'),
        'lora_adapter': dir_size_mb(REPO / 'models' / 'lora_adapter')}
    save_results(results)


_CACHE = {}


def _unpack(df):
    if 'data' not in _CACHE:
        _CACHE['data'] = transformer_data(df)
    return _CACHE['data']


# ------------------------------------------------------------- BERTopic (07)
def step_clustering(df, results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from bertopic import BERTopic

    print('\n=== [07] BERTopic clustering ===')
    docs = df.text.tolist()
    topic_model = BERTopic(embedding_model='all-MiniLM-L6-v2',
                           min_topic_size=30, nr_topics='auto', verbose=True)
    topics, _ = topic_model.fit_transform(docs)
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    print(f'Discovered {n_topics} topics (excluding outliers)')

    info = topic_model.get_topic_info()
    info.to_csv(REPO / 'docs_local' / 'bertopic_topic_info.csv', index=False)

    cross = pd.crosstab(pd.Series(df.failure_category, name='true_category'),
                        pd.Series(topics, name='bertopic_cluster'))
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cross, annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_title('True failure category vs. discovered BERTopic clusters')
    plt.tight_layout()
    plt.savefig(REPO / 'figures' / 'bertopic_vs_labels.png', dpi=150)
    plt.close()
    print('[saved] figures/bertopic_vs_labels.png')

    top_topics = [
        {'topic': int(r.Topic), 'count': int(r.Count), 'name': r.Name}
        for r in info.head(10).itertuples()]
    results['bertopic'] = {'n_topics': n_topics, 'top_topics': top_topics}
    save_results(results)


# ------------------------------------------------------ embeddings (08 prep)
def step_embeddings(df, results):
    from src.nlp_pipeline import embed, save_embeddings

    print('\n=== Regenerating sentence-transformer embeddings index ===')
    texts = df.text.tolist()
    embeddings = embed(texts)
    save_embeddings(embeddings, texts, str(REPO / 'models'))
    results['embeddings'] = {'shape': list(embeddings.shape),
                             'model': 'all-MiniLM-L6-v2'}
    save_results(results)


# ----------------------------------------------------------- similarity (08)
def step_similarity(df, results):
    from src.nlp_pipeline import cosine_similarity_search, embed, load_embeddings

    print('\n=== [08] Similarity search sanity check ===')
    embeddings, texts = load_embeddings(str(REPO / 'models'))
    query = ('High vibration alarm on pump P-104, bearing wear suspected, '
             'lubrication schedule may be inadequate')
    q_emb = embed([query], show_progress=False)
    top = cosine_similarity_search(q_emb[0], embeddings, top_k=3)
    print(f'Query: {query}')
    examples = []
    for rank, (idx, score) in enumerate(top, 1):
        text = texts[idx]
        print(f'  {rank}. ({score:.3f}) {text[:120]}')
        examples.append({'rank': rank, 'score': round(score, 4),
                         'text': text[:300]})
    results['similarity_demo'] = {'query': query, 'top3': examples}
    save_results(results)


# ----------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', default='tfidf,distilbert,lora,clustering,embeddings,similarity')
    parser.add_argument('--fast', action='store_true', help='2 epochs instead of 4')
    args = parser.parse_args()
    epochs = 2 if args.fast else 4
    steps = [s.strip() for s in args.steps.split(',')]

    df = pd.read_csv(REPO / 'data' / 'work_orders.csv')
    print(f'Loaded {len(df):,} work orders')
    results = load_results()
    results['run_meta'] = {'date': time.strftime('%Y-%m-%d %H:%M'),
                           'epochs': epochs, 'steps': steps}

    t_start = time.time()
    if 'tfidf' in steps:
        step_tfidf(df, results)
    if 'distilbert' in steps:
        step_distilbert(df, results, epochs)
    if 'lora' in steps:
        step_lora(df, results, epochs)
    if 'clustering' in steps:
        step_clustering(df, results)
    if 'embeddings' in steps:
        step_embeddings(df, results)
    if 'similarity' in steps:
        step_similarity(df, results)

    print(f'\nALL DONE in {(time.time() - t_start) / 60:.1f} min')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
