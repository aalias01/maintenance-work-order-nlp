"""ETL extraction evaluation (notebook 02 as a script) — rule-based vs. LLM vs. hybrid.

Resumable: per-record predictions are cached in docs_local/etl_cache.json, so the
script can be interrupted and re-run; it only processes records not yet cached.

    python scripts/run_etl_eval.py --mode rule_based
    python scripts/run_etl_eval.py --mode llm --budget-seconds 30
    python scripts/run_etl_eval.py --mode hybrid --budget-seconds 30
    python scripts/run_etl_eval.py --report

Evaluation: per-field accuracy with substring match, computed only on records
where the field is actually present in the text (ground truth not null) —
extractors are never penalized for unextractable fields. Coverage (fraction of
records where the field exists) is reported alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd
from dotenv import load_dotenv

load_dotenv(REPO / '.env')

from src.etl_extractor import ETLExtractor

SAMPLE_SIZE = 100
SEED = 42
FIELDS = ['equipment_tag', 'failure_mode', 'parts_replaced', 'root_cause',
          'failure_category']
CACHE_PATH = REPO / 'docs_local' / 'etl_cache.json'
RESULTS_PATH = REPO / 'docs_local' / 'run_results.json'


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.parent.mkdir(exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache))


def get_sample():
    df = pd.read_csv(REPO / 'data' / 'work_orders.csv')
    gt = json.loads((REPO / 'data' / 'work_orders_ground_truth.json').read_text())
    gt_df = pd.DataFrame(gt).set_index('work_order_id')
    sample = df.sample(n=SAMPLE_SIZE, random_state=SEED)
    return sample, gt_df


def run_mode(mode: str, budget_seconds: float) -> None:
    sample, _ = get_sample()
    cache = load_cache()
    cache.setdefault(mode, {})
    ext = ETLExtractor(mode=mode)
    t0 = time.time()
    done = 0
    for row in sample.itertuples():
        if row.work_order_id in cache[mode]:
            continue
        if time.time() - t0 > budget_seconds:
            break
        pred = ext.extract(row.text)
        cache[mode][row.work_order_id] = {f: getattr(pred, f) for f in FIELDS}
        cache[mode][row.work_order_id]['extractor_used'] = pred.extractor_used
        done += 1
        if done % 10 == 0:
            save_cache(cache)
    save_cache(cache)
    total = len(cache[mode])
    print(f'[{mode}] {total}/{SAMPLE_SIZE} records cached '
          f'(+{done} this run, {time.time() - t0:.0f}s)')
    if total >= SAMPLE_SIZE:
        print(f'[{mode}] COMPLETE')


def report() -> None:
    sample, gt_df = get_sample()
    cache = load_cache()
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else {}
    etl_results = {}
    for mode, preds in cache.items():
        if len(preds) < SAMPLE_SIZE:
            print(f'[{mode}] incomplete ({len(preds)}/{SAMPLE_SIZE}) — skipping')
            continue
        mode_scores = {}
        for field in FIELDS:
            correct = evaluable = 0
            for row in sample.itertuples():
                raw = gt_df.loc[row.work_order_id, field] \
                    if row.work_order_id in gt_df.index else None
                if raw is None or (isinstance(raw, float) and pd.isna(raw)):
                    continue  # field not extractable from this record
                evaluable += 1
                true_val = str(raw).lower().strip()
                pred_val = (preds[row.work_order_id].get(field) or '').lower().strip()
                if true_val in pred_val:
                    correct += 1
            mode_scores[field] = {
                'accuracy': round(correct / evaluable, 4) if evaluable else None,
                'evaluable': evaluable}
        if mode in ('llm', 'hybrid'):
            used = [p.get('extractor_used', '') for p in preds.values()]
            mode_scores['_llm_call_fraction'] = round(
                sum(1 for u in used if 'llm' in u) / len(used), 3)
        etl_results[mode] = mode_scores
        print(f'\n=== {mode} (n={SAMPLE_SIZE} sample) ===')
        for f in FIELDS:
            s = mode_scores[f]
            print(f"  {f:18s} {s['accuracy']:.1%}  (evaluable: {s['evaluable']})")
        if '_llm_call_fraction' in mode_scores:
            print(f"  LLM call fraction: {mode_scores['_llm_call_fraction']:.1%}")
    results['etl'] = etl_results
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))
    print(f'\n[saved] {RESULTS_PATH.relative_to(REPO)}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['rule_based', 'llm', 'hybrid'])
    parser.add_argument('--budget-seconds', type=float, default=600)
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.mode:
        run_mode(args.mode, args.budget_seconds)
    if args.report:
        report()
    if not args.mode and not args.report:
        parser.print_help()


if __name__ == '__main__':
    main()
