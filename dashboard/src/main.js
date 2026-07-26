// Athlete Intelligence -- Dashboard data & presentation layer.
//
// Every chart and metric on this page is driven exclusively by real data
// from the FastAPI backend's `/api/replay/*` endpoints, which replay a
// selected WSD4FEDSRM subject's actual recordings through this project's
// real preprocessing, feature engineering, model, SHAP, and personalization
// pipelines (see backend/services/replay_engine.py). Nothing here generates
// or approximates a signal -- there is no Math.random()/Math.sin() anywhere
// in this file. This file only changes HOW real values are labeled,
// formatted, and animated -- every fetch call, endpoint, and payload field
// below is unchanged from the underlying data-integration layer. The
// light/dark theme toggle is a pure display preference and never affects
// what data is fetched or how it's computed.

const API_BASE = 'http://127.0.0.1:8000';
const RING_CIRCUMFERENCE = 326.7; // 2 * PI * r(52), matches the SVG rings in index.html

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let isPlaying = true;
let currentEventSource = null;
let currentSubjectId = null;
let lastPayload = null;
let lastDecisionNarrative = null;
let lastPriorityLabel = null;
const fatigueHistory = []; // rolling {step, score, rpe}
const MAX_HISTORY_POINTS = 60;

const charts = {};
const counterAnimations = new WeakMap();

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

// ---------------------------------------------------------------------------
// Plain-language translation of the real, technical EMG feature names.
// These are exactly the 12 features the trained model + SHAP explainer use
// (ml/feature_engineering/emg_features.py); nothing here is invented, this
// only supplies a friendlier label + one-line description for each.
// ---------------------------------------------------------------------------

const FEATURE_TRANSLATIONS = {
  mav: { label: 'Average Muscle Effort', desc: 'How hard the muscle worked on average during this moment.' },
  rms: { label: 'Muscle Activity Level', desc: "The muscle's overall signal strength right now." },
  variance: { label: 'Effort Variability', desc: 'How much the muscle effort fluctuated moment to moment.' },
  waveform_length: { label: 'Movement Complexity', desc: 'How complex or erratic the muscle signal shape is.' },
  zero_crossings: { label: 'Signal Fluctuation Rate', desc: 'How rapidly the muscle signal changes direction.' },
  slope_sign_changes: { label: 'Muscle Firing Pattern', desc: 'A marker of how the muscle fibers are firing.' },
  willison_amplitude: { label: 'Muscle Contraction Bursts', desc: 'How many strong contraction bursts occurred.' },
  integrated_emg: { label: 'Total Muscle Engagement', desc: 'The cumulative muscle engagement over this moment.' },
  mean_frequency: { label: 'Muscle Fatigue Signal', desc: 'A frequency-based marker that shifts as muscles fatigue.' },
  median_frequency: { label: 'Muscle Fiber Fatigue Index', desc: 'Another frequency marker sensitive to fatigue.' },
  peak_frequency: { label: 'Dominant Muscle Frequency', desc: 'The strongest frequency in the muscle signal.' },
  spectral_entropy: { label: 'Signal Complexity', desc: 'How spread out vs. concentrated the signal energy is.' },
};

function translateFeature(featureKey) {
  return FEATURE_TRANSLATIONS[featureKey] || { label: featureKey, desc: 'A real signal feature used by the AI model.' };
}

// ---------------------------------------------------------------------------
// Small presentation helpers: animated counters + progress rings
// ---------------------------------------------------------------------------

function animateCounter(el, toValue, { decimals = 0, suffix = '' } = {}) {
  if (!el) return;
  const fromValue = parseFloat(el.dataset.rawValue || '0') || 0;
  el.dataset.rawValue = String(toValue);

  // Set the real final value synchronously first -- requestAnimationFrame is
  // throttled or fully paused by the browser while this tab/pane isn't being
  // composited (e.g. backgrounded), so the animation below must be a bonus
  // effect layered on top of a value that is already correct, never the only
  // path to displaying it.
  el.textContent = `${toValue.toFixed(decimals)}${suffix}`;

  const existing = counterAnimations.get(el);
  if (existing) cancelAnimationFrame(existing);

  const duration = 500;
  const start = performance.now();

  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = fromValue + (toValue - fromValue) * eased;
    el.textContent = `${current.toFixed(decimals)}${suffix}`;
    if (t < 1) {
      counterAnimations.set(el, requestAnimationFrame(tick));
    } else {
      counterAnimations.delete(el);
    }
  }
  counterAnimations.set(el, requestAnimationFrame(tick));
}

function setRingProgress(el, percent) {
  if (!el) return;
  const clamped = clamp(percent, 0, 100);
  const offset = RING_CIRCUMFERENCE * (1 - clamped / 100);
  el.style.strokeDashoffset = String(offset);
}

function scoreTone(score) {
  if (score >= 70) return { tone: 'tone-good', label: 'Low Fatigue' };
  if (score >= 40) return { tone: 'tone-moderate', label: 'Moderate Fatigue' };
  return { tone: 'tone-alert', label: 'High Fatigue' };
}

// ---------------------------------------------------------------------------
// Connection / error status
// ---------------------------------------------------------------------------

function setConnectionStatus(state, text) {
  const el = document.getElementById('connectionStatus');
  const textEl = document.getElementById('connectionStatusText');
  el.classList.remove('status-error', 'status-loading');
  if (state === 'error') el.classList.add('status-error');
  if (state === 'loading') el.classList.add('status-loading');
  textEl.textContent = text;
}

function showError(message) {
  const banner = document.getElementById('errorBanner');
  banner.textContent = message;
  banner.style.display = 'block';
}

function clearError() {
  const banner = document.getElementById('errorBanner');
  banner.style.display = 'none';
  banner.textContent = '';
}

// ---------------------------------------------------------------------------
// Chart initialization (empty datasets -- populated only from real payloads)
// ---------------------------------------------------------------------------

// Reads the current theme (light/dark) and returns the color set Chart.js
// should render with -- purely a rendering palette, never a data value.
function getChartPalette() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return isLight
    ? {
        text: '#4a5672', muted: '#7c869e', grid: 'rgba(15,23,42,0.06)',
        tooltipBg: 'rgba(255,255,255,0.97)', tooltipBorder: 'rgba(8,145,178,0.3)', tooltipTitle: '#101a2e', tooltipBody: '#4a5672',
        cyan: '#0891b2', blue: '#2563eb', purple: '#7c3aed', amber: '#b45309', crimson: '#dc2626',
        cyanFillStrong: 'rgba(8,145,178,0.22)', cyanFillWeak: 'rgba(8,145,178,0.0)',
        cyanTint: 'rgba(8,145,178,0.6)', cyanTintStrong: 'rgba(8,145,178,0.75)', cyanRadarFill: 'rgba(8,145,178,0.12)',
        neutralBar: 'rgba(15,23,42,0.18)', pointBorder: '#ffffff',
      }
    : {
        text: '#8fa3c4', muted: '#5b6c8f', grid: 'rgba(120,190,255,0.05)',
        tooltipBg: 'rgba(8,13,26,0.92)', tooltipBorder: 'rgba(46,230,255,0.35)', tooltipTitle: '#eaf4ff', tooltipBody: '#8fa3c4',
        cyan: '#2ee6ff', blue: '#4f8dff', purple: '#a855f7', amber: '#ffb545', crimson: '#ff3d5e',
        cyanFillStrong: 'rgba(46,230,255,0.3)', cyanFillWeak: 'rgba(46,230,255,0.0)',
        cyanTint: 'rgba(46,230,255,0.65)', cyanTintStrong: 'rgba(46,230,255,0.8)', cyanRadarFill: 'rgba(46,230,255,0.14)',
        neutralBar: 'rgba(120,145,190,0.4)', pointBorder: '#04070d',
      };
}

function initCharts() {
  const p = getChartPalette();
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400, easing: 'easeOutCubic' },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: p.text, font: { family: 'Space Grotesk', size: 11 }, usePointStyle: true, boxWidth: 8 } },
      tooltip: {
        backgroundColor: p.tooltipBg,
        borderColor: p.tooltipBorder,
        borderWidth: 1,
        titleColor: p.tooltipTitle,
        titleFont: { family: 'Space Grotesk', weight: '700' },
        bodyColor: p.tooltipBody,
        bodyFont: { family: 'Space Grotesk' },
        padding: 10,
        cornerRadius: 10,
        displayColors: true,
      },
    },
    scales: {
      x: { grid: { color: p.grid }, ticks: { color: p.muted, font: { family: 'Space Grotesk', size: 10 } } },
      y: { grid: { color: p.grid }, ticks: { color: p.muted, font: { family: 'Space Grotesk', size: 10 } } },
    },
  };

  function gradientFill(ctx, colorStart, colorEnd) {
    const g = ctx.createLinearGradient(0, 0, 0, 250);
    g.addColorStop(0, colorStart);
    g.addColorStop(1, colorEnd);
    return g;
  }

  charts.emg = new Chart(document.getElementById('emgChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Ant. Deltoid', data: [], borderColor: p.cyan, borderWidth: 2, pointRadius: 0, tension: 0.35 },
        { label: 'Infraspinatus', data: [], borderColor: p.purple, borderWidth: 2, pointRadius: 0, tension: 0.35 },
        { label: 'Lat. Dorsi', data: [], borderColor: p.amber, borderWidth: 2, pointRadius: 0, tension: 0.35 },
      ],
    },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'mV', color: p.muted } } } },
  });

  charts.imu = new Chart(document.getElementById('imuChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Acc X', data: [], borderColor: p.crimson, borderWidth: 2, pointRadius: 0, tension: 0.35 },
        { label: 'Acc Y', data: [], borderColor: p.blue, borderWidth: 2, pointRadius: 0, tension: 0.35 },
        { label: 'Acc Z', data: [], borderColor: p.amber, borderWidth: 2, pointRadius: 0, tension: 0.35 },
      ],
    },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'm/s²', color: p.muted } } } },
  });

  charts.ppg = new Chart(document.getElementById('ppgChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'Pulse Wave', data: [], borderColor: p.cyan, borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true,
        backgroundColor: (context) => gradientFill(context.chart.ctx, p.cyanFillStrong, p.cyanFillWeak),
      }],
    },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, title: { display: true, text: 'V', color: p.muted } } } },
  });

  charts.progression = new Chart(document.getElementById('fatigueProgressionChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Performance Score', data: [], borderColor: p.cyan, borderWidth: 2.5, pointRadius: 0, tension: 0.35, yAxisID: 'y' },
        { label: 'Recorded Borg RPE (self-report)', data: [], borderColor: p.muted, borderDash: [5, 5], borderWidth: 2, pointRadius: 2, yAxisID: 'y1' },
      ],
    },
    options: {
      ...commonOptions,
      scales: {
        x: commonOptions.scales.x,
        y: { position: 'left', min: 0, max: 100, title: { display: true, text: 'Performance Score', color: p.muted }, grid: commonOptions.scales.y.grid, ticks: commonOptions.scales.y.ticks },
        y1: { position: 'right', min: 6, max: 20, title: { display: true, text: 'Borg RPE', color: p.muted }, grid: { display: false }, ticks: commonOptions.scales.y.ticks },
      },
    },
  });

  charts.spectral = new Chart(document.getElementById('muscleSpectralChart').getContext('2d'), {
    type: 'bar',
    data: { labels: [], datasets: [{ label: 'Current value (raw)', data: [], backgroundColor: p.cyanTint, borderRadius: 6, maxBarThickness: 34 }] },
    options: commonOptions,
  });

  charts.shapBar = new Chart(document.getElementById('shapBarChart').getContext('2d'), {
    type: 'bar',
    indexAxis: 'y',
    data: { labels: [], datasets: [{ label: 'Overall importance', data: [], backgroundColor: p.cyan, borderRadius: 6 }] },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, grid: { display: false } } } },
  });

  charts.shapWaterfall = new Chart(document.getElementById('shapWaterfallChart').getContext('2d'), {
    type: 'bar',
    indexAxis: 'y',
    data: { labels: [], datasets: [{ label: 'Effect on this moment\'s score', data: [], backgroundColor: (ctx) => (ctx.raw > 0 ? p.crimson : p.cyan), borderRadius: 6 }] },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, grid: { display: false } } } },
  });

  charts.loso = new Chart(document.getElementById('losoComparisonChart').getContext('2d'), {
    type: 'bar',
    data: { labels: ['Population Average', 'You (Personalized)'], datasets: [{ label: 'Fatigue risk (%)', data: [0, 0], backgroundColor: [p.neutralBar, p.cyanTintStrong], borderRadius: 8, maxBarThickness: 70 }] },
    options: { ...commonOptions, scales: { ...commonOptions.scales, y: { ...commonOptions.scales.y, min: 0, max: 100 } } },
  });

  // Skill radar -- 4 real, independently-sourced metrics normalized to a
  // common 0-100 scale for display only (see updatePersonalizationTab for
  // the exact normalization of each axis). No fabricated axis.
  charts.skillRadar = new Chart(document.getElementById('skillRadarChart').getContext('2d'), {
    type: 'radar',
    data: {
      labels: ['Strength', 'Resilience', 'Focus', 'Endurance'],
      datasets: [{
        label: 'Skill Profile',
        data: [0, 0, 0, 0],
        backgroundColor: p.cyanRadarFill,
        borderColor: p.cyan,
        pointBackgroundColor: p.cyan,
        pointBorderColor: p.pointBorder,
        borderWidth: 2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 500, easing: 'easeOutCubic' },
      plugins: { legend: { display: false }, tooltip: commonOptions.plugins.tooltip },
      scales: {
        r: {
          min: 0,
          max: 100,
          angleLines: { color: p.grid },
          grid: { color: p.grid },
          pointLabels: { color: p.text, font: { family: 'Space Grotesk', size: 10 } },
          ticks: { display: false, stepSize: 25 },
        },
      },
    },
  });
}

// Rebuilds every Chart.js instance with the current theme's palette and
// replays the last real payload into them -- Chart.js datasets don't follow
// CSS variables, so a theme switch destroys and recreates the charts rather
// than leaving stale dark-mode colors on a light background.
function restyleChartsForTheme() {
  if (!charts.emg) return;
  Object.values(charts).forEach((c) => c && c.destroy());
  Object.keys(charts).forEach((k) => delete charts[k]);
  initCharts();
  if (lastPayload) updateAllTabs(lastPayload);
}

// ---------------------------------------------------------------------------
// Subject list + replay stream
// ---------------------------------------------------------------------------

async function loadSubjects() {
  const select = document.getElementById('subjectSelect');
  try {
    const response = await fetch(`${API_BASE}/api/replay/subjects`);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    select.innerHTML = '';
    data.subjects.forEach((subjectId) => {
      const option = document.createElement('option');
      option.value = subjectId;
      option.textContent = `Athlete ${subjectId}`;
      select.appendChild(option);
    });
    clearError();
    if (data.subjects.length > 0) {
      select.value = data.subjects[0];
      startReplayStream(data.subjects[0]);
    }
  } catch (err) {
    setConnectionStatus('error', 'Backend unavailable');
    showError(`Could not load the athlete list from the backend (${err.message}). Start it with: uvicorn backend.api.app:app --reload`);
    select.innerHTML = '<option value="">Backend unavailable</option>';
  }
}

function startReplayStream(subjectId) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  currentSubjectId = subjectId;
  fatigueHistory.length = 0;
  lastDecisionNarrative = null;
  lastPriorityLabel = null;
  resetCoachFeed();
  const contentBody = document.querySelector('.content-body');
  if (contentBody) {
    contentBody.style.opacity = '0';
    contentBody.style.transition = 'opacity 0.25s ease';
    requestAnimationFrame(() => { contentBody.style.opacity = '1'; });
  }
  setConnectionStatus('loading', `Loading ${subjectId}'s session (first load can take up to a minute)...`);

  const source = new EventSource(`${API_BASE}/api/replay/stream/${subjectId}?speed_ms=600`);
  currentEventSource = source;

  source.onopen = () => {
    setConnectionStatus('connected', `Live session streaming for ${subjectId}`);
    clearError();
  };

  source.onmessage = (event) => {
    if (!isPlaying) return;
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (err) {
      console.error('Malformed replay payload', err);
      return;
    }
    lastPayload = payload;
    clearError();
    updateAllTabs(payload);
  };

  source.onerror = () => {
    setConnectionStatus('error', 'Backend unavailable -- live stream disconnected');
    showError('Lost connection to the backend replay stream. Confirm the API is running: uvicorn backend.api.app:app --reload');
  };
}

// ---------------------------------------------------------------------------
// Per-tab real-data updates
// ---------------------------------------------------------------------------

function updateAllTabs(payload) {
  updateHero(payload);
  updateSignalsTab(payload);
  updatePredictionTab(payload);
  updateExplainabilityTab(payload);
  updatePersonalizationTab(payload);
  document.getElementById('triggerAgentCheck').disabled = false;
  updateRpeVsThreshold();
}

function mean(arr) {
  if (!arr || arr.length === 0) return null;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function updateHero(payload) {
  const probabilityPct = Math.round(payload.prediction.probability_fatigued * 100);
  const score = 100 - probabilityPct;

  document.getElementById('heroAthleteId').textContent = payload.subject_id;

  animateCounter(document.getElementById('heroScoreValue'), score, { decimals: 0 });
  setRingProgress(document.getElementById('heroRingProgress'), score);

  const { tone, label } = scoreTone(score);
  const pill = document.getElementById('heroStatusPill');
  pill.textContent = label;
  pill.className = `status-pill ${tone}`;
  document.getElementById('heroFatigueSub').textContent = `${probabilityPct}% fatigue risk · ${payload.direction.replace('_', ' ')}`;

  // The recommendation slot only ever shows a real AI Coach output (from the
  // last real check-in); it never fabricates a recommendation from raw
  // prediction data alone.
  if (lastDecisionNarrative) {
    document.getElementById('heroRecommendation').textContent = lastDecisionNarrative;
    document.getElementById('heroRecommendationSub').textContent = lastPriorityLabel ? `Priority: ${lastPriorityLabel}` : 'From your latest AI Coach check-in';
  }
}

function updateSignalsTab(payload) {
  const emg = payload.signals.emg.channels;
  const emgLabels = Array.from({ length: (emg.anterior_deltoid || []).length }, (_, i) => i);
  charts.emg.data.labels = emgLabels;
  charts.emg.data.datasets[0].data = emg.anterior_deltoid || [];
  charts.emg.data.datasets[1].data = emg.infraspinatus || [];
  charts.emg.data.datasets[2].data = emg.latissimus_dorsi || [];
  charts.emg.update('none');

  const imu = payload.signals.imu;
  const imuLabels = Array.from({ length: (imu.acc_x || []).length }, (_, i) => i);
  charts.imu.data.labels = imuLabels;
  charts.imu.data.datasets[0].data = imu.acc_x || [];
  charts.imu.data.datasets[1].data = imu.acc_y || [];
  charts.imu.data.datasets[2].data = imu.acc_z || [];
  charts.imu.update('none');

  const ppg = payload.signals.ppg;
  const ppgValues = ppg ? ppg.values : [];
  charts.ppg.data.labels = Array.from({ length: ppgValues.length }, (_, i) => i);
  charts.ppg.data.datasets[0].data = ppgValues;
  charts.ppg.update('none');

  const rmsEntry = payload.explainability.local_top_features.find((f) => f.feature === 'rms');
  document.getElementById('emgRmsVal').innerHTML = rmsEntry
    ? `${rmsEntry.raw_value.toFixed(4)} <span class="unit">(rms)</span>`
    : '-- <span class="unit"></span>';

  const accZMean = mean(imu.acc_z);
  document.getElementById('imuJerkVal').innerHTML = accZMean !== null ? `${accZMean.toFixed(2)} <span class="unit">m/s²</span>` : '--';

  const ppgMean = mean(ppgValues);
  document.getElementById('ppgHrVal').innerHTML = ppgMean !== null ? `${ppgMean.toFixed(3)} <span class="unit">V (mean)</span>` : '--';

  document.getElementById('conditionBadge').textContent = `Load ${payload.intensity_band}%`;
  document.getElementById('conditionDirection').textContent = payload.direction.replace('_', ' ');
  document.getElementById('conditionIdText').textContent = payload.condition_id;
}

function updatePredictionTab(payload) {
  const pred = payload.prediction;
  const probabilityPct = Math.round(pred.probability_fatigued * 100);
  const score = 100 - probabilityPct;

  animateCounter(document.getElementById('currentProbability'), score, { decimals: 0 });
  setRingProgress(document.getElementById('scoreRingProgress'), score);

  document.getElementById('fatigueRiskText').textContent = `${probabilityPct}%`;
  document.getElementById('predictedClassText').textContent = pred.predicted_label;
  document.getElementById('confidenceText').textContent = `${(pred.confidence * 100).toFixed(1)}%`;
  document.getElementById('estimatedRpeText').textContent = `${pred.estimated_rpe_proxy} / 20`;

  const rpe = payload.ground_truth.rpe_nearest;
  document.getElementById('groundTruthRpeText').textContent = rpe !== null ? `${rpe.toFixed(1)} / 20` : 'n/a for this moment';
  document.getElementById('currentTaskText').textContent = `${payload.direction} (${payload.intensity_band}% MVIC)`;

  const { tone, label } = scoreTone(score);
  const badge = document.getElementById('predictedLabelBadge');
  badge.textContent = label;
  badge.className = tone === 'tone-alert' ? 'rpe-status-label danger' : 'rpe-status-label';

  fatigueHistory.push({ step: payload.step, score, rpe });
  if (fatigueHistory.length > MAX_HISTORY_POINTS) fatigueHistory.shift();
  charts.progression.data.labels = fatigueHistory.map((h) => h.step);
  charts.progression.data.datasets[0].data = fatigueHistory.map((h) => h.score);
  charts.progression.data.datasets[1].data = fatigueHistory.map((h) => h.rpe);
  charts.progression.update('none');

  const features = payload.explainability.local_top_features;
  charts.spectral.data.labels = features.map((f) => translateFeature(f.feature).label);
  charts.spectral.data.datasets[0].data = features.map((f) => f.raw_value);
  charts.spectral.update('none');
}

function updateExplainabilityTab(payload) {
  const local = payload.explainability.local_top_features.slice().sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));
  const maxAbsShap = Math.max(...local.map((f) => Math.abs(f.shap)), 0.0001);

  const setInsightRow = (idx, feature) => {
    const titleEl = document.getElementById(`shapTopFeature${idx}`);
    const descEl = document.getElementById(`shapTopFeature${idx}Desc`);
    const impactEl = document.getElementById(`shapTopFeature${idx}Val`);
    const barEl = document.getElementById(`shapTopFeature${idx}Bar`);
    if (!feature) {
      titleEl.textContent = '--';
      descEl.textContent = 'Waiting for live data...';
      impactEl.textContent = '--';
      impactEl.className = 'insight-impact';
      barEl.style.width = '0%';
      return;
    }
    const t = translateFeature(feature.feature);
    titleEl.textContent = t.label;
    descEl.textContent = t.desc;
    const increasing = feature.shap >= 0;
    impactEl.textContent = increasing ? 'Increasing risk' : 'Reducing risk';
    impactEl.className = `insight-impact ${increasing ? 'up' : 'down'}`;
    barEl.className = `insight-bar-fill ${increasing ? 'up' : 'down'}`;
    barEl.style.width = `${Math.round((Math.abs(feature.shap) / maxAbsShap) * 100)}%`;
  };

  setInsightRow(1, local[0]);
  setInsightRow(2, local[1]);
  setInsightRow(3, local[2]);
  document.getElementById('shapConfidence').textContent = `${(payload.prediction.confidence * 100).toFixed(1)}%`;

  const techDetails = document.getElementById('shapTechDetails');
  techDetails.textContent = local
    .map((f) => `${f.feature}: ${f.shap >= 0 ? '+' : ''}${f.shap.toFixed(4)} SHAP  (raw value ${f.raw_value.toFixed(4)})`)
    .join('\n');

  const global = payload.explainability.global_top_features;
  charts.shapBar.data.labels = global.map((f) => translateFeature(f.feature).label);
  charts.shapBar.data.datasets[0].data = global.map((f) => f.importance);
  charts.shapBar.update('none');

  charts.shapWaterfall.data.labels = local.map((f) => translateFeature(f.feature).label);
  charts.shapWaterfall.data.datasets[0].data = local.map((f) => f.shap);
  charts.shapWaterfall.update('none');
}

// Reference max used only to normalize MVIC force (N) onto a 0-100 display
// axis for the skill radar / power ring -- a presentation-layer scale, not
// a clinical threshold.
const MVIC_DISPLAY_MAX_N = 150;

function updatePersonalizationTab(payload) {
  const profile = payload.personalization.profile;
  document.getElementById('profileSubjectId').textContent = payload.subject_id;
  document.getElementById('playerCardName').textContent = `Athlete ${payload.subject_id}`;

  let dominantHandLabel = '--';
  if (profile.demographic) {
    document.getElementById('profileSexAge').textContent = `${profile.demographic.sex} / ${profile.demographic.age} years`;
    document.getElementById('profileHand').textContent = profile.demographic.dominant_hand;
    dominantHandLabel = profile.demographic.dominant_hand
      ? profile.demographic.dominant_hand.charAt(0).toUpperCase() + profile.demographic.dominant_hand.slice(1)
      : '--';
    document.getElementById('profileCardArm').textContent = dominantHandLabel;
    document.getElementById('profileCardDemo').textContent = `${profile.demographic.sex}, ${profile.demographic.age}y`;
    document.getElementById('playerMetaArm').textContent = dominantHandLabel;
    document.getElementById('playerMetaDemo').textContent = `${profile.demographic.sex}, ${profile.demographic.age}y`;
  }
  if (profile.anthropometric) {
    document.getElementById('profileHeightMass').textContent = profile.demographic
      ? `${profile.demographic.height_cm} cm / ${profile.anthropometric.mass_kg} kg`
      : `${profile.anthropometric.mass_kg} kg`;
    document.getElementById('profileBmiFat').textContent = `${profile.anthropometric.bmi.toFixed(1)} / ${profile.anthropometric.body_fat_percent}%`;
    document.getElementById('profileCardBody').textContent = profile.anthropometric.bmi.toFixed(1);
  }

  let strengthPct = 0;
  if (profile.mvic) {
    document.getElementById('profileMvicIr').textContent = `${profile.mvic.ir_mean_n.toFixed(1)} N`;
    document.getElementById('profileMvicEr').textContent = `${profile.mvic.er_mean_n.toFixed(1)} N`;
    document.getElementById('profileCardStrength').textContent = `${profile.mvic.ir_mean_n.toFixed(0)} / ${profile.mvic.er_mean_n.toFixed(0)} N`;
    document.getElementById('playerMetaStrength').textContent = `${profile.mvic.ir_mean_n.toFixed(0)} / ${profile.mvic.er_mean_n.toFixed(0)} N`;
    strengthPct = clamp(((profile.mvic.ir_mean_n + profile.mvic.er_mean_n) / 2 / MVIC_DISPLAY_MAX_N) * 100, 0, 100);
  }

  const generic = Math.round(payload.personalization.generic_probability * 100);
  const personalized = Math.round(payload.personalization.personalized_probability * 100);
  charts.loso.data.datasets[0].data = [generic, personalized];
  charts.loso.update('none');

  // Skill radar + Power Level ring: four independently-sourced real values,
  // each normalized to 0-100 for a common display scale only.
  //   Strength   = average MVIC (IR/ER) vs. MVIC_DISPLAY_MAX_N
  //   Resilience = this moment's real Performance Score (100 - fatigue risk %)
  //   Focus      = the model's own real confidence for this prediction
  //   Endurance  = real recorded Borg RPE (self-report) inverted onto 0-100
  //                when available for this window, else falls back to
  //                Resilience so the radar never shows a fabricated value.
  const resiliencePct = 100 - Math.round(payload.prediction.probability_fatigued * 100);
  const focusPct = payload.prediction.confidence * 100;
  const groundTruthRpe = payload.ground_truth.rpe_nearest;
  const endurancePct = groundTruthRpe !== null && groundTruthRpe !== undefined
    ? clamp(100 - (groundTruthRpe / 20) * 100, 0, 100)
    : resiliencePct;

  charts.skillRadar.data.datasets[0].data = [strengthPct, resiliencePct, focusPct, endurancePct];
  charts.skillRadar.update('none');

  const powerLevel = Math.round((strengthPct + resiliencePct + focusPct + endurancePct) / 4);
  animateCounter(document.getElementById('playerPowerValue'), powerLevel, { decimals: 0 });
  setRingProgress(document.getElementById('playerPowerRing'), powerLevel);

  const tagEl = document.getElementById('playerCardTag');
  if (powerLevel >= 75) tagEl.textContent = 'Elite Tier Performer';
  else if (powerLevel >= 50) tagEl.textContent = 'Competitive Tier';
  else tagEl.textContent = 'Building Tier';
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function positionNavIndicator(activeBtn) {
  const indicator = document.getElementById('navSlideIndicator');
  if (!indicator || !activeBtn) return;
  indicator.style.transform = `translateY(${activeBtn.offsetTop}px)`;
  indicator.style.height = `${activeBtn.offsetHeight}px`;
}

function initNavigation() {
  const tabs = document.querySelectorAll('.nav-item');
  const pageTitle = document.getElementById('pageTitle');
  const pageSubtitle = document.getElementById('pageSubtitle');

  const titles = {
    signals: { title: 'Live Performance', sub: 'Real muscle, movement, and heart-response signals, streamed live from your last session' },
    predictions: { title: 'Performance Score', sub: 'A live, human-friendly read on how your body is handling this session' },
    explainability: { title: 'AI Insights', sub: "Why did the AI reach this conclusion? Here's what mattered most" },
    personalization: { title: 'Athlete Profile', sub: 'Real demographic, strength, and baseline data personalizing every prediction' },
    agents: { title: 'AI Coach', sub: 'Your real-time coaching assistant, powered by the live agent backend' },
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      positionNavIndicator(tab);
      document.querySelectorAll('.tab-content').forEach((content) => content.classList.remove('active'));
      const activeEl = document.getElementById(`tab-${tab.dataset.tab}`);
      if (activeEl) activeEl.classList.add('active');
      if (titles[tab.dataset.tab]) {
        pageTitle.textContent = titles[tab.dataset.tab].title;
        pageSubtitle.textContent = titles[tab.dataset.tab].sub;
      }
    });
  });

  const initialActive = document.querySelector('.nav-item.active') || tabs[0];
  positionNavIndicator(initialActive);
  window.addEventListener('resize', () => positionNavIndicator(document.querySelector('.nav-item.active')));
}

function initPlaybackToggle() {
  const btn = document.getElementById('togglePlayback');
  const text = document.getElementById('playbackBtnText');
  const iconWrap = document.getElementById('playbackIconWrap');
  const pausedBadge = document.getElementById('pausedBadge');

  btn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    text.textContent = isPlaying ? 'Pause Stream' : 'Resume Stream';
    iconWrap.innerHTML = `<i data-lucide="${isPlaying ? 'pause' : 'play'}" style="width: 14px; height: 14px;"></i>`;
    btn.classList.toggle('is-paused', !isPlaying);
    pausedBadge.style.display = isPlaying ? 'none' : 'inline-block';
    refreshIcons();
  });
}

function initSubjectSelector() {
  document.getElementById('subjectSelect').addEventListener('change', (e) => {
    if (e.target.value) startReplayStream(e.target.value);
  });
}

// ---------------------------------------------------------------------------
// Theme toggle (dark / light) -- a display preference only. Persisted in
// localStorage so it survives a reload; never touches the replay stream,
// the subject selection, or any fetched data.
// ---------------------------------------------------------------------------

const THEME_STORAGE_KEY = 'athlete-intelligence-theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);

  // Lucide converts <i data-lucide> into a rendered <svg> the first time
  // createIcons() runs; simply updating data-lucide on that svg and calling
  // createIcons() again does not re-render it. Swapping in a fresh <i>
  // placeholder guarantees the icon actually changes.
  const iconWrap = document.getElementById('themeToggleIconWrap');
  const label = document.getElementById('themeToggleText');
  if (iconWrap) iconWrap.innerHTML = `<i data-lucide="${theme === 'light' ? 'sun' : 'moon'}" style="width: 14px; height: 14px;"></i>`;
  if (label) label.textContent = theme === 'light' ? 'Light' : 'Dark';
  refreshIcons();

  restyleChartsForTheme();
}

function initThemeToggle() {
  const saved = localStorage.getItem(THEME_STORAGE_KEY) === 'light' ? 'light' : 'dark';
  applyTheme(saved);

  document.getElementById('themeToggle').addEventListener('click', () => {
    const current = document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
    applyTheme(current === 'light' ? 'dark' : 'light');
  });
}

// ---------------------------------------------------------------------------
// AI Coach -- real backend only, no local fallback simulation. The primary
// output is a structured Recommendation Card built directly from the real,
// already-computed fields on DecisionAndReportingAgent's result
// (backend/agents/decision_agent.py: action_required, recommendation) --
// nothing here is inferred or reworded by the frontend. Supporting agent
// narratives appear underneath as short, calm cards. Avatars are abstract
// Lucide glyphs (radio tower / dna / lightbulb / brain circuit), never a
// human icon.
// ---------------------------------------------------------------------------

// Maps the backend's real `action_required` value to display language.
// CONTINUE_MONITORING / REST_RECOMMENDED / IMMEDIATE_TERMINATION are the
// only three values DecisionAndReportingAgent ever produces.
const PRIORITY_DISPLAY = {
  CONTINUE_MONITORING: { label: 'Normal', tone: '', badgeTone: 'badge-success', recovery: 'Recovering Well' },
  REST_RECOMMENDED: { label: 'Elevated', tone: 'tone-alert', badgeTone: 'badge-moderate', recovery: 'Rest Advised' },
  IMMEDIATE_TERMINATION: { label: 'Critical', tone: 'tone-alert', badgeTone: 'badge-alert', recovery: 'Stop & Recover' },
};

// Badge ids for the "Meet Your AI Team" roster -- reflect the real, already
// -returned `success` / `warnings` / `action_required` fields from each
// agent's result (backend/api/app.py already exposes these; the frontend
// previously just never displayed them).
const AGENT_BADGE_IDS = {
  data_agent: 'dataAgentBadge',
  personalization_agent: 'personalizationAgentBadge',
  explainability_agent: 'explainabilityAgentBadge',
  decision_agent: 'decisionAgentBadge',
};

function setAgentBadge(agentKey, text, tone) {
  const el = document.getElementById(AGENT_BADGE_IDS[agentKey]);
  if (!el) return;
  el.textContent = text;
  el.className = `badge ${tone}`;
}

function resetAgentBadges() {
  Object.keys(AGENT_BADGE_IDS).forEach((key) => setAgentBadge(key, 'Ready', 'badge-success'));
}

function resetCoachFeed() {
  const feed = document.getElementById('agentLogs');
  if (feed) feed.innerHTML = '<div class="coach-placeholder">Select an athlete and click "Check In With Coach" to get a live assessment.</div>';
  const slot = document.getElementById('recommendationCardSlot');
  if (slot) slot.innerHTML = '';
  resetAgentBadges();
  document.getElementById('triggerAgentCheck').disabled = true;
  document.getElementById('currentRpeVsThreshold').textContent = 'Waiting for live data...';
}

function addCoachBubble({ icon, name, text, thinking = false }) {
  const feed = document.getElementById('agentLogs');
  const placeholder = feed.querySelector('.coach-placeholder');
  if (placeholder) placeholder.remove();

  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const bubble = document.createElement('div');
  bubble.className = `coach-bubble ${thinking ? 'is-thinking' : ''}`;
  bubble.innerHTML = `
    <div class="coach-avatar"><i data-lucide="${icon}"></i></div>
    <div class="coach-body">
      <div class="coach-meta">
        <span class="coach-agent-name">${name}</span>
        <span class="coach-timestamp">${time}</span>
      </div>
      <div class="coach-text">${text}</div>
    </div>
  `;
  feed.prepend(bubble);
  refreshIcons();
  return bubble;
}

// Live comparison of this moment's real RPE-proxy reading against the
// coach's alert threshold -- both values are already displayed elsewhere
// on the dashboard, just not side by side where the sensitivity slider
// makes the comparison meaningful.
function updateRpeVsThreshold() {
  const el = document.getElementById('currentRpeVsThreshold');
  if (!el || !lastPayload) return;
  const current = lastPayload.prediction.estimated_rpe_proxy;
  const threshold = parseFloat(document.getElementById('alertThreshold').value);
  const cls = current >= threshold ? 'over-threshold' : 'under-threshold';
  el.innerHTML = `Current reading: <span class="${cls}">${current.toFixed(1)}</span> · Threshold: ${threshold.toFixed(1)}`;
}

function renderRecommendationCard(decisionData, confidencePct) {
  const slot = document.getElementById('recommendationCardSlot');
  const info = PRIORITY_DISPLAY[decisionData.action_required] || { label: decisionData.action_required, tone: '', recovery: '--' };
  lastPriorityLabel = info.label;

  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  slot.innerHTML = `
    <div class="recommendation-card ${info.tone} margin-top">
      <div class="recommendation-header">
        <h4><i data-lucide="sparkles"></i> Coach Recommendation</h4>
        <span class="coach-timestamp">${time}</span>
      </div>
      <div class="recommendation-fields">
        <div><span class="rf-label">Priority</span><span class="rf-value">${info.label}</span></div>
        <div><span class="rf-label">Confidence</span><span class="rf-value">${confidencePct}%</span></div>
        <div><span class="rf-label">Recovery Outlook</span><span class="rf-value">${info.recovery}</span></div>
      </div>
      <div class="recommendation-action"><strong>Suggested Action:</strong> ${decisionData.recommendation}</div>
    </div>
  `;
  refreshIcons();
  slot.querySelector('.recommendation-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Turns each agent's real `success` (and, for the Signal Analyst, real
// `warnings_count`) into a roster badge -- these fields already come back
// from POST /api/agents/analyze today; nothing new is requested from the
// backend.
function badgeForDataAgent(res) {
  if (!res.success) return { text: 'Issue Detected', tone: 'badge-alert' };
  const warnings = res.data && res.data.warnings_count;
  if (warnings) return { text: 'Warnings', tone: 'badge-moderate' };
  return { text: 'Certified', tone: 'badge-success' };
}

function badgeForGenericAgent(res) {
  return res.success ? { text: 'Complete', tone: 'badge-success' } : { text: 'Issue', tone: 'badge-alert' };
}

function initAgentLoop() {
  const triggerBtn = document.getElementById('triggerAgentCheck');
  const slider = document.getElementById('alertThreshold');
  const threshVal = document.getElementById('thresholdVal');

  slider.addEventListener('input', (e) => {
    threshVal.textContent = `${e.target.value} (Borg RPE proxy threshold)`;
    updateRpeVsThreshold();
  });

  triggerBtn.addEventListener('click', async () => {
    if (!lastPayload) {
      addCoachBubble({ icon: 'alert-triangle', name: 'Coach', text: 'No live data yet -- wait for the stream to connect before checking in.' });
      return;
    }

    const threshold = parseFloat(slider.value);
    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Checking in...';
    Object.keys(AGENT_BADGE_IDS).forEach((key) => setAgentBadge(key, 'Analyzing…', 'badge-neutral'));
    const thinkingBubble = addCoachBubble({
      icon: 'loader-circle',
      name: 'AI Coach',
      text: '<span class="thinking-dots"><span></span><span></span><span></span></span> Reviewing your live data...',
      thinking: true,
    });

    try {
      const response = await fetch(`${API_BASE}/api/agents/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_id: lastPayload.subject_id, step: lastPayload.step, threshold }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const result = await response.json();
      const pipeline = result.agent_pipeline_results;
      thinkingBubble.remove();

      if (pipeline.decision_agent) {
        const confidencePct = (lastPayload.prediction.confidence * 100).toFixed(1);
        renderRecommendationCard(pipeline.decision_agent.data, confidencePct);
        lastDecisionNarrative = pipeline.decision_agent.data.recommendation;
        if (lastPayload) updateHero(lastPayload);
        const info = PRIORITY_DISPLAY[pipeline.decision_agent.data.action_required];
        setAgentBadge('decision_agent', info ? info.label : pipeline.decision_agent.data.action_required, info ? info.badgeTone : 'badge-neutral');
      }
      if (pipeline.data_agent) {
        addCoachBubble({ icon: 'radio-tower', name: 'Signal Analyst', text: pipeline.data_agent.narrative });
        const b = badgeForDataAgent(pipeline.data_agent);
        setAgentBadge('data_agent', b.text, b.tone);
      }
      if (pipeline.personalization_agent) {
        addCoachBubble({ icon: 'dna', name: 'Personalization Coach', text: pipeline.personalization_agent.narrative });
        const b = badgeForGenericAgent(pipeline.personalization_agent);
        setAgentBadge('personalization_agent', b.text, b.tone);
      }
      if (pipeline.explainability_agent) {
        addCoachBubble({ icon: 'lightbulb', name: 'Insight Translator', text: pipeline.explainability_agent.narrative });
        const b = badgeForGenericAgent(pipeline.explainability_agent);
        setAgentBadge('explainability_agent', b.text, b.tone);
      }
      clearError();
    } catch (err) {
      thinkingBubble.remove();
      addCoachBubble({ icon: 'alert-triangle', name: 'Coach', text: `Backend unavailable -- the check-in could not run (${err.message}).` });
      showError('AI Coach check-in failed: the FastAPI backend is unreachable. No simulated result was shown.');
      Object.keys(AGENT_BADGE_IDS).forEach((key) => setAgentBadge(key, 'Unavailable', 'badge-alert'));
    } finally {
      triggerBtn.disabled = false;
      triggerBtn.textContent = 'Check In With Coach';
    }
  });
}

// ---------------------------------------------------------------------------
// Document Ready
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initCharts();
  initNavigation();
  initPlaybackToggle();
  initAgentLoop();
  initSubjectSelector();
  loadSubjects();
  refreshIcons();
});
