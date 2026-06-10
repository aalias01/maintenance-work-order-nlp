'use strict';

// Live API on Render (free tier sleeps when idle — first request may take ~60 s).
// For local development, change to 'http://localhost:8000'.
const API_BASE = 'https://maintenance-nlp-api.onrender.com';

const CATEGORY_LABELS = {
  mechanical_failure:      'Mechanical Failure',
  electrical_failure:      'Electrical Failure',
  hydraulic_failure:       'Hydraulic Failure',
  instrumentation_failure: 'Instrumentation Failure',
  preventive_maintenance:  'Preventive Maintenance',
  operator_damage:         'Operator Damage',
};

const EXAMPLES = {
  mech: "Responded to high vibration alarm on P-104 (centrifugal pump). Investigation found bearing wear on inboard bearing. Root cause: inadequate lubrication over extended run time. Replaced mechanical seal and bearing set. Aligned shaft coupling and returned to service. Equipment operating satisfactory at design conditions.",
  elec: "WO raised for M-210 induction motor. Operator reported equipment not starting from DCS. Found motor tripped on overcurrent — phase C reading 0 amps. Burning smell noted from terminal box. Replaced burned motor windings — sent to motor shop for rewind. Verified operation after reinstall — all three phases balanced.",
  inst: "DCS reading 245 psi on pressure transmitter PT-322 while gauge reads 185 psi — 60 psi deviation. Impulse line found plugged with process buildup. Flushed impulse lines and replaced pressure transmitter. Calibrated against reference gauge — verified within 2 psi. Cleared alarm.",
  pm:   "Completed annual PM on compressor C-101 per schedule. Changed compressor oil and filter. Replaced V-belt set and coupling insert. Lubricated all grease points per lube route card. No anomalies noted during inspection. All measurements within specification. Unit returned to service.",
};

function labelForCategory(category) {
  return CATEGORY_LABELS[category] || category.replace(/_/g, ' ');
}

function appendText(tagName, className, text, parent) {
  const el = document.createElement(tagName);
  if (className) el.className = className;
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

// ─── Health check ─────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = document.getElementById('api-status');
  try {
    const resp = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(4000) });
    const data = await resp.json();
    const statusText = data.classifier_loaded
      ? `✅ API ready · ${data.model_mode} · ${data.corpus_size.toLocaleString()} records`
      : '⚠️ API degraded — run notebooks first';
    badge.textContent = statusText;
    badge.className = `status-badge ${data.classifier_loaded ? 'status-ok' : 'status-error'}`;
  } catch {
    badge.textContent = '❌ API offline';
    badge.className = 'status-badge status-error';
  }
}

// ─── Examples ─────────────────────────────────────────────────────────────────
function loadExample(key) {
  document.getElementById('wo-text').value = EXAMPLES[key] || '';
}

// ─── Classification ───────────────────────────────────────────────────────────
async function runClassification() {
  const btn = document.getElementById('classify-btn');
  const text = document.getElementById('wo-text').value.trim();
  if (!text) { alert('Please enter a work order description.'); return; }

  btn.disabled = true;
  btn.textContent = 'Classifying…';

  try {
    const resp = await fetch(`${API_BASE}/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || `HTTP ${resp.status}`);
    }
    const data = await resp.json();
    renderResult(data);
  } catch (e) {
    alert(`Classification failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Classify & Find Similar Cases';
  }
}

// ─── Render ───────────────────────────────────────────────────────────────────
function renderResult(data) {
  document.getElementById('result-section').classList.remove('hidden');

  // Category badge
  const badge = document.getElementById('category-badge');
  badge.textContent = labelForCategory(data.category);
  badge.className = `category-badge cat-${data.category}`;

  // Confidence
  document.getElementById('confidence-val').textContent = `${(data.confidence * 100).toFixed(1)}%`;

  // Score bars
  const scoresEl = document.getElementById('all-scores');
  scoresEl.innerHTML = '';
  const sortedCats = Object.entries(data.all_scores).sort((a, b) => b[1] - a[1]);
  for (const [cat, score] of sortedCats) {
    const pct = Math.round(score * 100);
    const isBest = cat === data.category;
    const row = document.createElement('div');
    row.className = 'score-row';
    appendText('div', 'score-name', cat.replace(/_/g, ' '), row);
    const track = document.createElement('div');
    track.className = 'score-bar-track';
    const fill = document.createElement('div');
    fill.className = `score-bar-fill ${isBest ? 'best' : ''}`;
    fill.style.width = `${pct}%`;
    track.appendChild(fill);
    row.appendChild(track);
    appendText('div', 'score-val', `${pct}%`, row);
    scoresEl.appendChild(row);
  }

  // Extracted fields
  const efEl = document.getElementById('extracted-fields');
  if (data.extracted_fields) {
    efEl.classList.remove('hidden');
    const ftEl = document.getElementById('fields-table');
    ftEl.innerHTML = '';
    for (const [key, val] of Object.entries(data.extracted_fields)) {
      if (!val) continue;
      const row = document.createElement('div');
      row.className = 'fields-row';
      appendText('span', 'field-key', key.replace(/_/g, ' '), row);
      appendText('span', 'field-val', String(val), row);
      ftEl.appendChild(row);
    }
  } else {
    efEl.classList.add('hidden');
  }

  // Similar cases
  const listEl = document.getElementById('similar-list');
  listEl.innerHTML = '';
  if (data.similar_cases && data.similar_cases.length > 0) {
    for (const c of data.similar_cases) {
      const sim = Math.round(c.similarity_score * 100);
      const item = document.createElement('div');
      item.className = 'similar-item';
      const header = document.createElement('div');
      header.className = 'similar-header';
      appendText('span', 'similar-id', c.work_order_id, header);
      appendText('span', 'similar-score', `${sim}% similar`, header);
      item.appendChild(header);
      appendText(
        'span',
        `similar-category cat-${c.failure_category}`,
        labelForCategory(c.failure_category),
        item,
      );
      appendText('p', 'similar-text', c.text, item);
      listEl.appendChild(item);
    }
  } else {
    const message = appendText('p', null, 'Similarity search not available - build embeddings index in notebook 08.', listEl);
    message.className = 'empty-state';
  }

  document.getElementById('result-section').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// ─── Init ─────────────────────────────────────────────────────────────────────
checkHealth();
