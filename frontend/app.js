'use strict';

const API_BASE = 'https://maintenance-nlp-api.onrender.com';

const CATEGORY_LABELS = {
  mechanical_failure: 'Mechanical failure',
  electrical_failure: 'Electrical failure',
  hydraulic_failure: 'Hydraulic failure',
  instrumentation_failure: 'Instrumentation failure',
  preventive_maintenance: 'Preventive maintenance',
  operator_damage: 'Operator damage',
};

const SAMPLES = [
  {
    text: 'Responded to high vibration alarm on P-104 (centrifugal pump). Investigation found bearing wear on inboard bearing. Root cause: inadequate lubrication over extended run time. Replaced mechanical seal and bearing set. Aligned shaft coupling and returned to service. Equipment operating satisfactory at design conditions.',
    intended: 'mechanical_failure',
    provenance: 'authored',
  },
  {
    text: 'WO raised for M-210 induction motor. Operator reported equipment not starting from DCS. Found motor tripped on overcurrent, phase C reading 0 amps. Burning smell noted from terminal box. Replaced burned motor windings, sent to motor shop for rewind. Verified operation after reinstall, all three phases balanced.',
    intended: 'electrical_failure',
    provenance: 'authored',
  },
  {
    text: 'DCS reading 245 psi on pressure transmitter PT-322 while gauge reads 185 psi, 60 psi deviation. Impulse line found plugged with process buildup. Flushed impulse lines and replaced pressure transmitter. Calibrated against reference gauge, verified within 2 psi. Cleared alarm.',
    intended: 'instrumentation_failure',
    provenance: 'authored',
  },
  {
    text: 'Completed annual PM on compressor C-101 per schedule. Changed compressor oil and filter. Replaced V-belt set and coupling insert. Lubricated all grease points per lube route card. No anomalies noted during inspection. All measurements within specification. Unit returned to service.',
    intended: 'preventive_maintenance',
    provenance: 'authored',
  },
  {
    text: 'HPU-3 supplying press line lost pressure during shift, gauge reading 1450 psi against 2200 psi setpoint. Found supply hose to cylinder C-12 weeping at the crimp fitting and reservoir low. Replaced hose assembly, topped off reservoir, bled trapped air at cylinder. Pressure held at setpoint through test cycles. RTS.',
    intended: 'hydraulic_failure',
    provenance: 'authored',
  },
  {
    text: 'Forklift contacted conveyor CV-7 leg guard while staging pallets. Guard bent into belt path and rubbing. No injury reported. Straightened leg guard, replaced two anchor bolts, checked belt tracking and alignment. Reviewed staging clearance with shift lead. Returned to service same day.',
    intended: 'operator_damage',
    provenance: 'authored',
  },
  {
    text: 'Space heater circuit for control room panel tripping breaker CB-14 repeatedly. Meggered circuit, found heating element shorted to frame. Replaced element and inspected wiring for heat damage. Breaker holding under load after replacement.',
    intended: 'electrical_failure',
    provenance: 'authored',
  },
  {
    text: 'Quarterly PM on exhaust fan EF-9 per route. Greased bearings per lube chart, checked belt tension and sheave wear, verified amp draw within nameplate. Slight belt glazing noted, belt replaced. All readings logged, unit returned to service.',
    intended: 'preventive_maintenance',
    provenance: 'authored',
  },
];

const state = {
  sampleIndex: 0,
  activeSample: null,
  healthStarted: 0,
  healthWarmup: null,
  runWarmup: null,
};

const els = {
  html: document.documentElement,
  toggle: document.getElementById('mode-toggle'),
  apiStatus: document.getElementById('api-status'),
  nameplateWarmup: document.getElementById('nameplate-warmup'),
  runWarmup: document.getElementById('run-warmup'),
  orderText: document.getElementById('order-text'),
  provenance: document.getElementById('provenance-line'),
  sampleButton: document.getElementById('sample-button'),
  manualButton: document.getElementById('manual-button'),
  manualRunButton: document.getElementById('manual-run-button'),
  helper: document.getElementById('control-helper'),
  error: document.getElementById('error-block'),
  log: document.getElementById('run-log'),
  category: document.getElementById('category-reading'),
  confidence: document.getElementById('confidence-reading'),
  scale: document.getElementById('confidence-scale'),
  intended: document.getElementById('intended-line'),
  scores: document.getElementById('all-scores'),
  fields: document.getElementById('fields-table'),
  similar: document.getElementById('similar-list'),
};

function labelForCategory(category) {
  return CATEGORY_LABELS[category] || String(category || '').replace(/_/g, ' ');
}

function setStoredMode(theme) {
  const mode = theme === 'night' ? 'dark' : 'light';
  const expires = new Date();
  expires.setFullYear(expires.getFullYear() + 1);
  document.cookie = `mode=${mode}; expires=${expires.toUTCString()}; path=/; domain=.alvinalias.com; SameSite=Lax`;
  localStorage.setItem('mode', mode);
}

function applyTheme(theme) {
  els.html.setAttribute('data-theme', theme);
  const night = theme === 'night';
  els.toggle.setAttribute('aria-pressed', String(night));
  els.toggle.setAttribute('aria-label', night ? 'Switch to light mode' : 'Switch to dark mode');
}

function toggleTheme() {
  const next = els.html.getAttribute('data-theme') === 'night' ? 'day' : 'night';
  applyTheme(next);
  setStoredMode(next);
}

function appendLog(line) {
  const row = document.createElement('div');
  row.textContent = line;
  els.log.appendChild(row);
  els.log.scrollTop = els.log.scrollHeight;
}

function clearError() {
  els.error.classList.add('hidden');
  els.error.textContent = '';
}

function showError(message, detail) {
  els.error.classList.remove('hidden');
  els.error.innerHTML = '';
  const text = document.createElement('p');
  text.textContent = message;
  els.error.appendChild(text);
  if (detail) {
    const raw = document.createElement('div');
    raw.className = 'error-detail';
    raw.textContent = detail;
    els.error.appendChild(raw);
  }
}

function formatStatus(data) {
  if (!data.classifier_loaded) return 'classifier not loaded';
  const parts = ['ready', data.model_mode || 'model'];
  if (Number(data.corpus_size) > 0) parts.push(`${Number(data.corpus_size).toLocaleString()} records`);
  return parts.join(' · ');
}

function confidenceSvg(value, red) {
  const width = 420;
  const height = 72;
  const x0 = 12;
  const x1 = 408;
  const y = 32;
  const clamped = Math.max(0, Math.min(1, Number(value) || 0));
  const marker = x0 + (x1 - x0) * clamped;
  let ticks = '';
  for (let i = 0; i <= 20; i += 1) {
    const x = x0 + ((x1 - x0) * i) / 20;
    const major = i % 5 === 0;
    ticks += `<line x1="${x}" y1="${y}" x2="${x}" y2="${y + (major ? 14 : 8)}" />`;
  }
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${clamped.toFixed(2)} confidence">
      <line class="scale-line" x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" />
      <g class="scale-ticks">${ticks}</g>
      <g class="scale-labels">
        <text x="${x0}" y="64">0</text>
        <text x="${(x0 + x1) / 2}" y="64" text-anchor="middle">0.5</text>
        <text x="${x1}" y="64" text-anchor="end">1</text>
      </g>
      <path class="scale-marker ${red ? 'red-marker' : ''}" d="M ${marker - 6} 11 L ${marker + 6} 11 L ${marker} 27 Z" />
    </svg>
  `;
}

function warmupSvg(seconds, overrun) {
  const width = 300;
  const height = 64;
  const x0 = 12;
  const x1 = 288;
  const y = 28;
  const markerSeconds = overrun ? 0 : Math.max(0, Math.min(60, seconds));
  const marker = x0 + (x1 - x0) * (1 - markerSeconds / 60);
  let ticks = '';
  for (let i = 0; i <= 12; i += 1) {
    const x = x0 + ((x1 - x0) * i) / 12;
    const major = i % 3 === 0;
    ticks += `<line x1="${x}" y1="${y}" x2="${x}" y2="${y + (major ? 12 : 7)}" />`;
  }
  return `
    <svg viewBox="0 0 ${width} ${height}" aria-hidden="true">
      <line class="scale-line" x1="${x0}" y1="${y}" x2="${x1}" y2="${y}" />
      <g class="scale-ticks">${ticks}</g>
      <g class="scale-labels">
        <text x="${x0}" y="58">0</text>
        <text x="${x0 + (x1 - x0) * 0.25}" y="58" text-anchor="middle">15</text>
        <text x="${x0 + (x1 - x0) * 0.5}" y="58" text-anchor="middle">30</text>
        <text x="${x0 + (x1 - x0) * 0.75}" y="58" text-anchor="middle">45</text>
        <text x="${x1}" y="58" text-anchor="end">60</text>
      </g>
      <path class="scale-marker ${overrun ? 'red-marker' : ''}" d="M ${marker - 5} 8 L ${marker + 5} 8 L ${marker} 23 Z" />
    </svg>
  `;
}

function startWarmup(container, compact) {
  const warmup = {
    remaining: 60,
    elapsed: 0,
    overrun: false,
    timer: null,
    container,
    compact,
  };

  function draw() {
    const number = warmup.overrun ? warmup.elapsed : warmup.remaining;
    const label = warmup.overrun ? 'seconds elapsed · still starting' : 'estimated seconds to warm';
    container.classList.remove('hidden');
    container.innerHTML = `
      <div class="warmup-number">${number}</div>
      <div class="warmup-main">
        ${warmupSvg(warmup.remaining, warmup.overrun)}
        <div class="warmup-label">${label}</div>
        <div class="warmup-log">&gt; warm-up estimate counting · this is an estimate, not progress</div>
      </div>
    `;
  }

  draw();
  warmup.timer = window.setInterval(() => {
    if (warmup.remaining > 0) {
      warmup.remaining -= 1;
    } else {
      if (!warmup.overrun) appendLog('> past the usual window · still waiting, counting up honestly');
      warmup.overrun = true;
      warmup.elapsed += 1;
    }
    draw();
  }, 1000);
  return warmup;
}

function finishWarmup(warmup, started) {
  if (!warmup) return;
  window.clearInterval(warmup.timer);
  const measured = ((performance.now() - started) / 1000).toFixed(1);
  warmup.container.innerHTML = `
    <div class="warmup-number">0</div>
    <div class="warmup-main">
      ${warmupSvg(0, false)}
      <div class="warmup-label">ready</div>
      <div class="warmup-log">&gt; awake · measured wake time ${measured} s</div>
    </div>
  `;
  appendLog(`> awake · measured wake time ${measured} s`);
  const remove = () => warmup.container.classList.add('hidden');
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    remove();
  } else {
    window.setTimeout(remove, 4000);
  }
}

function renderScores(scores, predicted) {
  els.scores.innerHTML = '';
  if (!scores) return;
  Object.entries(scores)
    .sort((a, b) => b[1] - a[1])
    .forEach(([category, score]) => {
      const row = document.createElement('div');
      row.className = 'score-row';
      const label = document.createElement('span');
      label.textContent = labelForCategory(category);
      const bar = document.createElement('span');
      bar.className = 'score-bar';
      const fill = document.createElement('span');
      fill.style.width = `${Math.max(0, Math.min(100, score * 100)).toFixed(1)}%`;
      if (category === predicted) fill.className = 'best';
      bar.appendChild(fill);
      const pct = document.createElement('span');
      pct.className = 'score-pct';
      pct.textContent = `${Math.round(score * 100)}%`;
      row.append(label, bar, pct);
      els.scores.appendChild(row);
    });
}

function renderFields(fields) {
  els.fields.innerHTML = '';
  const entries = Object.entries(fields || {}).filter(([, value]) => value);
  if (entries.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No fields returned.';
    els.fields.appendChild(empty);
    return;
  }
  entries.forEach(([key, value]) => {
    const row = document.createElement('div');
    row.className = 'field-row';
    const k = document.createElement('span');
    k.textContent = key.replace(/_/g, ' ');
    const v = document.createElement('span');
    v.textContent = String(value);
    row.append(k, v);
    els.fields.appendChild(row);
  });
}

function renderSimilar(cases) {
  els.similar.innerHTML = '';
  if (!cases || cases.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    empty.textContent = 'No similar cases returned.';
    els.similar.appendChild(empty);
    return;
  }
  cases.forEach((item) => {
    const row = document.createElement('article');
    row.className = 'similar-item';
    const pct = Math.round(Number(item.similarity_score || 0) * 100);
    const head = document.createElement('p');
    head.className = 'similar-head';
    head.textContent = `${item.work_order_id} · ${pct}% similar · ${labelForCategory(item.failure_category)}`;
    const text = document.createElement('p');
    text.className = 'similar-text';
    text.textContent = item.text || '';
    row.append(head, text);
    els.similar.appendChild(row);
  });
}

function renderResult(data, sample) {
  const predicted = data.category;
  const confidence = Number(data.confidence || 0);
  const predictedLabel = labelForCategory(predicted);
  const mismatch = Boolean(sample && sample.intended && sample.intended !== predicted);

  els.category.textContent = predictedLabel;
  els.category.title = 'API field: category';
  els.category.classList.toggle('earned-red', mismatch);
  els.confidence.textContent = `${confidence.toFixed(2)} confidence`;
  els.scale.innerHTML = confidenceSvg(confidence, mismatch);

  if (sample && sample.intended) {
    const intended = labelForCategory(sample.intended);
    els.intended.textContent = mismatch
      ? `Intended when written: ${intended} · the model read it as ${predictedLabel}`
      : `Intended when written: ${intended} · match`;
    els.intended.classList.toggle('earned-red', mismatch);
  } else {
    els.intended.textContent = 'your text · no intended label to compare';
    els.intended.classList.remove('earned-red');
  }

  renderScores(data.all_scores, predicted);
  renderFields(data.extracted_fields);
  renderSimilar(data.similar_cases);
}

function validateText(text) {
  if (text.length < 10) return 'Enter at least a sentence of work order text (10 characters or more).';
  if (text.length > 2000) return 'Keep it under 2,000 characters.';
  return '';
}

async function classifyCurrent(sample) {
  const text = els.orderText.value.trim();
  const validation = validateText(text);
  clearError();
  if (validation) {
    showError(validation);
    return;
  }

  const started = performance.now();
  let warmTimer = window.setTimeout(() => {
    appendLog('> server was asleep · sent the wake call');
    state.runWarmup = startWarmup(els.runWarmup, false);
  }, 2500);

  els.sampleButton.disabled = true;
  els.manualButton.disabled = true;
  els.manualRunButton.disabled = true;

  try {
    const response = await fetch(`${API_BASE}/classify`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });
    window.clearTimeout(warmTimer);
    warmTimer = null;
    finishWarmup(state.runWarmup, started);
    state.runWarmup = null;

    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try {
        const data = await response.json();
        detail = data.detail || detail;
      } catch {
        detail = response.statusText || detail;
      }
      throw new Error(detail);
    }

    const data = await response.json();
    appendLog(`> classified in ${((performance.now() - started) / 1000).toFixed(1)} s`);
    renderResult(data, sample);
  } catch (error) {
    window.clearTimeout(warmTimer);
    if (state.runWarmup) {
      window.clearInterval(state.runWarmup.timer);
      els.runWarmup.classList.add('hidden');
      state.runWarmup = null;
    }
    const isNetwork = error instanceof TypeError;
    const message = isNetwork
      ? 'Could not classify this text. Try again in a moment.'
      : `Could not classify this text. ${error.message}`;
    showError(message, error.message);
  } finally {
    els.sampleButton.disabled = false;
    els.manualButton.disabled = false;
    els.manualRunButton.disabled = false;
  }
}

function loadSample() {
  clearError();
  const index = state.sampleIndex % SAMPLES.length;
  const sample = SAMPLES[index];
  state.sampleIndex += 1;
  state.activeSample = sample;
  els.orderText.value = sample.text;
  els.provenance.textContent = `sample ${index + 1} of ${SAMPLES.length} · written by me, labeled ${labelForCategory(sample.intended)}`;
  els.intended.textContent = '';
  appendLog(`> sample ${index + 1} of ${SAMPLES.length} loaded · intended label hidden`);
  classifyCurrent(sample);
}

function enterManualMode() {
  state.activeSample = null;
  els.orderText.readOnly = false;
  els.orderText.value = '';
  els.provenance.textContent = 'your text · no intended label to compare';
  els.manualRunButton.classList.remove('hidden');
  els.orderText.focus();
  clearError();
}

function readManualText() {
  state.activeSample = null;
  els.provenance.textContent = 'your text · no intended label to compare';
  classifyCurrent(null);
}

async function checkHealth() {
  state.healthStarted = performance.now();
  const warmTimer = window.setTimeout(() => {
    state.healthWarmup = startWarmup(els.nameplateWarmup, true);
  }, 2500);

  try {
    const response = await fetch(`${API_BASE}/health`);
    window.clearTimeout(warmTimer);
    const data = await response.json();
    els.apiStatus.textContent = response.ok ? formatStatus(data) : 'classifier not loaded';
    finishWarmup(state.healthWarmup, state.healthStarted);
    state.healthWarmup = null;
  } catch {
    window.clearTimeout(warmTimer);
    els.apiStatus.textContent = 'server unreachable right now';
    if (state.healthWarmup) {
      window.clearInterval(state.healthWarmup.timer);
      state.healthWarmup.container.classList.add('hidden');
      state.healthWarmup = null;
    }
  }
}

function init() {
  applyTheme(els.html.getAttribute('data-theme') || 'day');
  els.scale.innerHTML = confidenceSvg(0, false);
  els.toggle.addEventListener('click', toggleTheme);
  els.sampleButton.addEventListener('click', loadSample);
  els.manualButton.addEventListener('click', enterManualMode);
  els.manualRunButton.addEventListener('click', readManualText);
  els.orderText.addEventListener('input', () => {
    state.activeSample = null;
    els.provenance.textContent = 'your text · no intended label to compare';
    els.manualRunButton.classList.remove('hidden');
  });
  checkHealth();
}

init();
