// Athlete Intelligence -- dashboard.
//
// Every figure shown here comes from the FastAPI backend's real endpoints
// (`/api/replay/*`, `/api/agents/analyze`, `/api/rag/explain`), which replay a
// chosen athlete's actual WSD4FEDSRM recording through this project's real
// preprocessing, models, SHAP explainer and agents. Nothing in this file
// generates or approximates data; if the backend is unavailable the page says
// so instead of showing anything made up.
//
// This file's job is translation: turning model output into plain words an
// athlete or coach can read ("Hard, 16 out of 20", "Higher than usual").

const API_BASE = 'http://127.0.0.1:8000';
const RING_CIRCUMFERENCE = 326.7; // 2 * PI * r(52), matches the SVG ring in index.html
const MAX_HISTORY_POINTS = 120;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let isPlaying = true;
let currentEventSource = null;
let lastPayload = null;
let lastAdvice = null; // { status, text } from the latest coach check-in
const effortHistory = []; // {t, rms} -- muscle effort per window
const ratingHistory = []; // {t, estimate, reported}

const charts = {};
const counterAnimations = new WeakMap();

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

// ---------------------------------------------------------------------------
// Plain-language translation
// ---------------------------------------------------------------------------

// The standard Borg 6-20 scale verbal anchors, which athletes and coaches know.
const BORG_ANCHORS = [
  [6, 'No effort'], [7, 'Extremely light'], [9, 'Very light'], [11, 'Light'],
  [13, 'Somewhat hard'], [15, 'Hard'], [17, 'Very hard'], [19, 'Extremely hard'], [20, 'Maximal'],
];

// A reading counts as a Borg word once it has reached that anchor (14.6 is still
// "Somewhat hard"; 15 is "Hard"). This keeps the words in step with the status
// thresholds below and with the coach's warning level of 15.
function effortWord(value) {
  return BORG_ANCHORS.reduce((best, anchor) => (anchor[0] <= value ? anchor : best), BORG_ANCHORS[0])[1];
}

// Status thresholds line up with the coach's own safety levels (15 = heavy
// fatigue warning, 18 = critical) in backend/agents/decision_agent.py.
function effortStatus(value) {
  if (value >= 18) return { word: 'Exhausted', tone: 'tone-alert' };
  if (value >= 15) return { word: 'Tiring', tone: 'tone-moderate' };
  if (value >= 11) return { word: 'Working', tone: 'tone-good' };
  return { word: 'Fresh', tone: 'tone-good' };
}

function effortLabel(value) {
  return `${effortWord(value)} (${value.toFixed(0)} out of 20)`;
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// z = how many of this athlete's own standard deviations a signal sits from
// their own average (from the backend's per-athlete baseline).
function usualWord(z) {
  if (z === undefined || z === null) return 'Not available';
  if (z >= 1) return 'Much higher than usual';
  if (z >= 0.4) return 'Higher than usual';
  if (z <= -1) return 'Much lower than usual';
  if (z <= -0.4) return 'Lower than usual';
  return 'About usual';
}

// Median frequency of the muscle signal drifts down as a muscle tires.
function tirednessSignWord(z) {
  if (z === undefined || z === null) return 'Not available';
  if (z <= -1) return 'Strong sign';
  if (z <= -0.4) return 'Slight sign';
  return 'No clear sign';
}

// Plain names and one-line meanings for the 12 real signal features the model uses.
const FEATURE_TRANSLATIONS = {
  mav: { label: 'Average muscle effort', desc: 'How hard the muscle worked on average.' },
  rms: { label: 'Muscle activity level', desc: "The muscle's overall signal strength." },
  variance: { label: 'Effort variability', desc: 'How much the effort fluctuated.' },
  waveform_length: { label: 'Movement complexity', desc: 'How busy or erratic the muscle signal is.' },
  zero_crossings: { label: 'Signal fluctuation rate', desc: 'How quickly the muscle signal changes direction.' },
  slope_sign_changes: { label: 'Muscle firing pattern', desc: 'How the muscle fibres are firing.' },
  willison_amplitude: { label: 'Muscle contraction bursts', desc: 'How many strong contractions happened.' },
  integrated_emg: { label: 'Total muscle engagement', desc: 'The total muscle activity in this moment.' },
  mean_frequency: { label: 'Muscle tiredness signal', desc: 'A signal that slows down as muscles tire.' },
  median_frequency: { label: 'Muscle fibre tiredness', desc: 'Another signal that slows down as muscles tire.' },
  peak_frequency: { label: 'Dominant muscle rhythm', desc: 'The strongest rhythm in the muscle signal.' },
  spectral_entropy: { label: 'Signal complexity', desc: 'How spread out the muscle signal is.' },
};

function translateFeature(key) {
  return FEATURE_TRANSLATIONS[key] || { label: key, desc: 'A signal feature used by the model.' };
}

// The effort figure the dashboard (and the coach) use for this moment: the
// out-of-sample Ridge estimate once ~30 s of the exercise have replayed,
// otherwise the classifier-probability proxy -- `source` always says which.
function currentRpeEstimate(payload) {
  const pred = payload.prediction;
  return pred.estimated_rpe || { value: pred.estimated_rpe_proxy, source: 'rf_probability_proxy' };
}

// The careful (Ridge) estimate needs ~30 s of an exercise. Until then the only
// figure is the fatigue detector's rough guess, which has not been validated as an
// effort estimate, so it is never used as the headline.
function estimateReady(payload) {
  const source = currentRpeEstimate(payload).source;
  const reason = (payload.prediction.borg_regression || {}).reason;
  return source === 'ridge_lopo' || reason === 'no_study_data';
}

function warmupSeconds(payload) {
  const from = (payload.prediction.borg_regression || {}).available_from_s;
  return from != null ? Math.round(from) : 30;
}

function describeEstimateMethod(payload) {
  const pred = payload.prediction;
  const borg = pred.borg_regression || {};
  if (currentRpeEstimate(payload).source === 'ridge_lopo') {
    return `Estimated by comparing this athlete's signals with ${borg.trained_on_athletes} other athletes. It is typically off by about ${borg.typical_error_borg} points, so treat it as a guide, not a measurement.`;
  }
  if (estimateReady(payload)) return 'A quick estimate from the fatigue detector (no comparison data is available for this athlete).';
  return `Warming up: the careful estimate starts after ${warmupSeconds(payload)} seconds of this exercise. Until then the fatigue detector's rough guess is ${effortLabel(currentRpeEstimate(payload).value).toLowerCase()}, which is not reliable on its own.`;
}

// ---------------------------------------------------------------------------
// Small presentation helpers
// ---------------------------------------------------------------------------

function animateCounter(el, toValue, { decimals = 0, suffix = '' } = {}) {
  if (!el) return;
  const fromValue = parseFloat(el.dataset.rawValue || '0') || 0;
  el.dataset.rawValue = String(toValue);

  // Set the real value synchronously first: requestAnimationFrame is paused for
  // backgrounded tabs, so the animation is a bonus, never the only path to the number.
  el.textContent = `${toValue.toFixed(decimals)}${suffix}`;

  const existing = counterAnimations.get(el);
  if (existing) cancelAnimationFrame(existing);

  const duration = 500;
  const start = performance.now();
  function tick(now) {
    const t = Math.min(1, (now - start) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    el.textContent = `${(fromValue + (toValue - fromValue) * eased).toFixed(decimals)}${suffix}`;
    if (t < 1) counterAnimations.set(el, requestAnimationFrame(tick));
    else counterAnimations.delete(el);
  }
  counterAnimations.set(el, requestAnimationFrame(tick));
}

function setRingProgress(el, percent) {
  if (!el) return;
  el.style.strokeDashoffset = String(RING_CIRCUMFERENCE * (1 - clamp(percent, 0, 100) / 100));
}

// ---------------------------------------------------------------------------
// Connection / error status
// ---------------------------------------------------------------------------

function setConnectionStatus(state, text) {
  const el = document.getElementById('connectionStatus');
  el.classList.remove('status-error', 'status-loading');
  if (state === 'error') el.classList.add('status-error');
  if (state === 'loading') el.classList.add('status-loading');
  setText('connectionStatusText', text);
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
// Charts (empty until real data arrives)
// ---------------------------------------------------------------------------

function getChartPalette() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return isLight
    ? {
        text: '#4a5672', muted: '#7c869e', grid: 'rgba(15,23,42,0.06)',
        tooltipBg: 'rgba(255,255,255,0.97)', tooltipBorder: 'rgba(8,145,178,0.3)', tooltipTitle: '#101a2e', tooltipBody: '#4a5672',
        cyan: '#0891b2', blue: '#2563eb', purple: '#7c3aed', amber: '#b45309', crimson: '#dc2626',
        cyanFillStrong: 'rgba(8,145,178,0.22)', cyanFillWeak: 'rgba(8,145,178,0.0)',
      }
    : {
        text: '#8fa3c4', muted: '#6b7c9e', grid: 'rgba(120,190,255,0.06)',
        tooltipBg: 'rgba(8,13,26,0.92)', tooltipBorder: 'rgba(46,230,255,0.35)', tooltipTitle: '#eaf4ff', tooltipBody: '#8fa3c4',
        cyan: '#2ee6ff', blue: '#4f8dff', purple: '#a855f7', amber: '#ffb545', crimson: '#ff3d5e',
        cyanFillStrong: 'rgba(46,230,255,0.3)', cyanFillWeak: 'rgba(46,230,255,0.0)',
      };
}

function initCharts() {
  const p = getChartPalette();
  const font = { family: 'Inter', size: 12 };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 300 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: p.text, font, usePointStyle: true, boxWidth: 8 } },
      tooltip: { backgroundColor: p.tooltipBg, borderColor: p.tooltipBorder, borderWidth: 1, titleColor: p.tooltipTitle, bodyColor: p.tooltipBody, padding: 10, cornerRadius: 10 },
    },
    scales: {
      x: { grid: { color: p.grid }, ticks: { color: p.muted, font, maxTicksLimit: 8 } },
      y: { grid: { color: p.grid }, ticks: { color: p.muted, font } },
    },
  };
  const fill = (ctx) => {
    const g = ctx.createLinearGradient(0, 0, 0, 230);
    g.addColorStop(0, p.cyanFillStrong);
    g.addColorStop(1, p.cyanFillWeak);
    return g;
  };
  const withYTitle = (text) => ({ ...common, scales: { ...common.scales, y: { ...common.scales.y, title: { display: true, text, color: p.muted, font } } } });

  // Plain-language charts
  charts.effortTrend = new Chart(document.getElementById('effortTrendChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Muscle effort', data: [], borderColor: p.cyan, borderWidth: 2.5, pointRadius: 0, tension: 0.35, fill: true, backgroundColor: (c) => fill(c.chart.ctx) }] },
    options: {
      ...common,
      plugins: { ...common.plugins, legend: { display: false }, tooltip: { ...common.plugins.tooltip, enabled: false } },
      scales: { x: { ...common.scales.x, title: { display: true, text: 'Seconds into the exercise', color: p.muted, font } }, y: { ...common.scales.y, min: 0, ticks: { display: false }, title: { display: true, text: 'Effort (relative)', color: p.muted, font } } },
    },
  });

  charts.progression = new Chart(document.getElementById('fatigueProgressionChart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        { label: 'Estimated effort', data: [], borderColor: p.cyan, borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
        { label: "Athlete's own rating", data: [], borderColor: p.amber, borderDash: [6, 4], borderWidth: 2, pointRadius: 3, spanGaps: true },
      ],
    },
    options: {
      ...common,
      scales: {
        x: { ...common.scales.x, title: { display: true, text: 'Seconds into the exercise', color: p.muted, font } },
        y: { ...common.scales.y, min: 6, max: 20, ticks: { ...common.scales.y.ticks, stepSize: 2 } },
      },
    },
  });

  // Technical raw-signal charts (shown only inside the collapsed "Technical details")
  charts.emg = new Chart(document.getElementById('emgChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Front shoulder muscle', data: [], borderColor: p.cyan, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
      { label: 'Back shoulder muscle', data: [], borderColor: p.purple, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
      { label: 'Back muscle', data: [], borderColor: p.amber, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
    ] },
    options: withYTitle('Muscle signal (mV)'),
  });
  charts.imu = new Chart(document.getElementById('imuChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'Movement, direction 1', data: [], borderColor: p.crimson, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
      { label: 'Movement, direction 2', data: [], borderColor: p.blue, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
      { label: 'Movement, direction 3', data: [], borderColor: p.amber, borderWidth: 1.5, pointRadius: 0, tension: 0.2 },
    ] },
    options: withYTitle('Arm movement (m/s²)'),
  });
  charts.ppg = new Chart(document.getElementById('ppgChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Pulse', data: [], borderColor: p.cyan, borderWidth: 2, pointRadius: 0, tension: 0.4, fill: true, backgroundColor: (c) => fill(c.chart.ctx) }] },
    options: withYTitle('Pulse signal (V)'),
  });
}

// Chart.js datasets don't follow CSS variables, so a theme switch rebuilds the
// charts with the new palette and refills them from the data already received.
function restyleChartsForTheme() {
  if (!charts.effortTrend) return;
  Object.values(charts).forEach((c) => c && c.destroy());
  Object.keys(charts).forEach((k) => delete charts[k]);
  initCharts();
  refreshHistoryCharts();
  if (lastPayload) updateRawSignalCharts(lastPayload);
}

function refreshHistoryCharts() {
  if (!charts.effortTrend) return;
  charts.effortTrend.data.labels = effortHistory.map((h) => Math.round(h.t));
  charts.effortTrend.data.datasets[0].data = effortHistory.map((h) => h.rms);
  charts.effortTrend.update('none');

  charts.progression.data.labels = ratingHistory.map((h) => Math.round(h.t));
  charts.progression.data.datasets[0].data = ratingHistory.map((h) => h.estimate);
  charts.progression.data.datasets[1].data = ratingHistory.map((h) => h.reported);
  charts.progression.update('none');
}

// ---------------------------------------------------------------------------
// Athlete list + live stream
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
    setConnectionStatus('error', 'Cannot reach the backend');
    showError(`The athlete list could not be loaded (${err.message}). Start the backend with: uvicorn backend.api.app:app --reload`);
    select.innerHTML = '<option value="">Backend unavailable</option>';
  }
}

function startReplayStream(subjectId) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  lastPayload = null;
  lastAdvice = null;
  effortHistory.length = 0;
  ratingHistory.length = 0;
  refreshHistoryCharts();
  resetCoachFeed();
  setText('heroRecommendation', 'Open the Coach tab and tap "Check in" for advice.');
  setText('heroRecommendationSub', '');

  const contentBody = document.querySelector('.content-body');
  if (contentBody) {
    contentBody.style.opacity = '0';
    contentBody.style.transition = 'opacity 0.25s ease';
    requestAnimationFrame(() => { contentBody.style.opacity = '1'; });
  }
  setConnectionStatus('loading', `Loading Athlete ${subjectId}. The first time can take a few minutes...`);

  const source = new EventSource(`${API_BASE}/api/replay/stream/${subjectId}?speed_ms=600`);
  currentEventSource = source;

  source.onopen = () => {
    setConnectionStatus('connected', `Live: Athlete ${subjectId}`);
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
    setConnectionStatus('error', 'Connection lost');
    showError('The live view lost its connection to the backend. Make sure it is running: uvicorn backend.api.app:app --reload');
  };
}

// ---------------------------------------------------------------------------
// Per-tab updates
// ---------------------------------------------------------------------------

function updateAllTabs(payload) {
  trackHistory(payload);
  updateHero(payload);
  updateRightNow(payload);
  updateHowTired(payload);
  updateWhy(payload);
  updateAboutYou(payload);
  document.getElementById('triggerAgentCheck').disabled = !estimateReady(payload);
  updateRpeVsThreshold();
}

function mean(arr) {
  return arr && arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : null;
}

function rmsOf(arr) {
  return arr && arr.length ? Math.sqrt(arr.reduce((a, b) => a + b * b, 0) / arr.length) : null;
}

function trackHistory(payload) {
  const t = payload.elapsed_seconds;
  const rms = rmsOf(payload.signals.emg.channels.anterior_deltoid);
  if (rms !== null) effortHistory.push({ t, rms });
  if (effortHistory.length > MAX_HISTORY_POINTS) effortHistory.shift();

  const reported = payload.ground_truth.rpe_nearest;
  ratingHistory.push({ t, estimate: estimateReady(payload) ? currentRpeEstimate(payload).value : null, reported });
  if (ratingHistory.length > MAX_HISTORY_POINTS) ratingHistory.shift();

  refreshHistoryCharts();
}

function updateHero(payload) {
  const value = currentRpeEstimate(payload).value;
  const pill = document.getElementById('heroStatusPill');
  const ring = document.getElementById('heroRingProgress');
  const valueEl = document.getElementById('heroScoreValue');

  setText('heroAthleteId', payload.subject_id);
  setText('heroActivity', ` · ${capitalize(payload.direction.replace('_', ' '))}`);

  if (estimateReady(payload)) {
    const status = effortStatus(value);
    animateCounter(valueEl, value, { decimals: 0 });
    setRingProgress(ring, ((value - 6) / 14) * 100);
    pill.textContent = status.word;
    pill.className = `status-pill ${status.tone}`;
    setText('heroEffortSub', `Effort is ${effortWord(value).toLowerCase()}, ${value.toFixed(0)} out of 20.`);
  } else {
    valueEl.dataset.rawValue = '0';
    valueEl.textContent = '--';
    setRingProgress(ring, 0);
    pill.textContent = 'Warming up';
    pill.className = 'status-pill tone-neutral';
    setText('heroEffortSub', `The careful reading starts after ${warmupSeconds(payload)} seconds of this exercise.`);
  }

  if (lastAdvice) {
    setText('heroRecommendation', lastAdvice.text);
    setText('heroRecommendationSub', `From your latest check-in: ${lastAdvice.status}`);
  } else if (estimateReady(payload)) {
    setText('heroRecommendation', 'Open the Coach tab and tap "Check in" for advice.');
    setText('heroRecommendationSub', '');
  } else {
    setText('heroRecommendation', 'The coach can give advice once the careful reading starts.');
    setText('heroRecommendationSub', '');
  }
}

function updateRightNow(payload) {
  const zscores = payload.personalization.feature_baseline_zscores || {};
  const seconds = Math.round(payload.elapsed_seconds);
  setText('activityText', `${capitalize(payload.direction.replace('_', ' '))} · ${payload.intensity_band}% of maximum strength · ${seconds} ${seconds === 1 ? 'second' : 'seconds'} in`);
  setText('tileEffortValue', usualWord(zscores.rms));
  setText('tileFatigueValue', tirednessSignWord(zscores.median_frequency));
  const reported = payload.ground_truth.rpe_nearest;
  setText('tileReportedValue', reported === null ? 'No rating right now' : effortLabel(reported));
  updateRawSignalCharts(payload);
}

function updateRawSignalCharts(payload) {
  if (!document.getElementById('rawSignalsDetails').open) return; // only draw what is visible
  const emg = payload.signals.emg.channels;
  const labels = (arr) => Array.from({ length: (arr || []).length }, (_, i) => i);
  charts.emg.data.labels = labels(emg.anterior_deltoid);
  charts.emg.data.datasets[0].data = emg.anterior_deltoid || [];
  charts.emg.data.datasets[1].data = emg.infraspinatus || [];
  charts.emg.data.datasets[2].data = emg.latissimus_dorsi || [];
  charts.emg.update('none');

  const imu = payload.signals.imu;
  charts.imu.data.labels = labels(imu.acc_x);
  charts.imu.data.datasets[0].data = imu.acc_x || [];
  charts.imu.data.datasets[1].data = imu.acc_y || [];
  charts.imu.data.datasets[2].data = imu.acc_z || [];
  charts.imu.update('none');

  const ppgValues = payload.signals.ppg ? payload.signals.ppg.values : [];
  charts.ppg.data.labels = labels(ppgValues);
  charts.ppg.data.datasets[0].data = ppgValues;
  charts.ppg.update('none');
}

function updateHowTired(payload) {
  const pred = payload.prediction;
  const value = currentRpeEstimate(payload).value;
  const reported = payload.ground_truth.rpe_nearest;

  const ready = estimateReady(payload);
  setText('effortHeadline', ready ? `${effortWord(value)}, ${value.toFixed(1)} out of 20` : `Warming up. The careful reading starts after ${warmupSeconds(payload)} seconds of this exercise.`);
  setText('estimatedRpeText', ready ? effortLabel(value) : 'Not ready yet');
  setText('groundTruthRpeText', reported === null ? 'No rating right now' : effortLabel(reported));
  setText('fatigueRiskText', `${Math.round(pred.probability_fatigued * 100)}%`);
  setText('estimatedRpeSource', describeEstimateMethod(payload));

  const toPercent = (v) => `${clamp(((v - 6) / 14) * 100, 0, 100)}%`;
  const estimateMarker = document.getElementById('meterEstimate');
  estimateMarker.style.display = ready ? 'block' : 'none';
  if (ready) estimateMarker.style.left = toPercent(value);
  const reportedMarker = document.getElementById('meterReported');
  reportedMarker.style.display = reported === null ? 'none' : 'block';
  if (reported !== null) reportedMarker.style.left = toPercent(reported);
}

function updateWhy(payload) {
  const local = payload.explainability.local_top_features.slice().sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap));

  const setRow = (idx, feature) => {
    if (!feature) {
      setText(`shapTopFeature${idx}`, '--');
      setText(`shapTopFeature${idx}Desc`, 'Waiting for live data...');
      setText(`shapTopFeature${idx}Val`, '--');
      return;
    }
    const t = translateFeature(feature.feature);
    const up = feature.shap >= 0;
    setText(`shapTopFeature${idx}`, t.label);
    setText(`shapTopFeature${idx}Desc`, t.desc);
    const chip = document.getElementById(`shapTopFeature${idx}Val`);
    chip.textContent = up ? 'Pushes the reading up' : 'Pulls the reading down';
    chip.className = `chip ${up ? 'chip-up' : 'chip-down'}`;
  };
  setRow(1, local[0]);
  setRow(2, local[1]);
  setRow(3, local[2]);

  setText('shapTechDetails', local.map((f) => `${f.feature}: ${f.shap >= 0 ? '+' : ''}${f.shap.toFixed(4)}  (value ${f.raw_value.toFixed(4)})`).join('\n'));
  setText('shapGlobalDetails', payload.explainability.global_top_features.map((f) => `${f.feature}: ${f.importance.toFixed(4)}`).join('\n'));
}

function updateAboutYou(payload) {
  const profile = payload.personalization.profile;
  setText('profileSubjectId', `Athlete ${payload.subject_id}`);

  if (profile.demographic) {
    const d = profile.demographic;
    setText('profileCardDemo', `${d.age} years · ${d.sex}`);
    setText('profileCardArm', d.dominant_hand ? capitalize(d.dominant_hand) : 'Not recorded');
  }
  if (profile.demographic && profile.anthropometric) {
    setText('profileHeightMass', `${profile.demographic.height_cm} cm · ${profile.anthropometric.mass_kg} kg`);
  }
  if (profile.mvic) {
    setText('profileCardStrength', `${profile.mvic.ir_mean_n.toFixed(0)} N in · ${profile.mvic.er_mean_n.toFixed(0)} N out`);
  }

  const h = profile.fatigue_history;
  if (h && h.mean_rpe !== null && h.mean_rpe !== undefined) {
    setText('historyText', `Across ${h.n_trials} earlier exercises, this athlete rated their effort between ${h.min_rpe.toFixed(0)} (${effortWord(h.min_rpe).toLowerCase()}) and ${h.max_rpe.toFixed(0)} (${effortWord(h.max_rpe).toLowerCase()}), averaging ${h.mean_rpe.toFixed(1)} out of 20.`);
  } else {
    setText('historyText', 'No earlier effort ratings are on record for this athlete.');
  }

  setText('genericRiskText', `${Math.round(payload.personalization.generic_probability * 100)}%`);
  setText('personalRiskText', `${Math.round(payload.personalization.personalized_probability * 100)}%`);
}

// ---------------------------------------------------------------------------
// Navigation, playback, theme
// ---------------------------------------------------------------------------

function positionNavIndicator(activeBtn) {
  const indicator = document.getElementById('navSlideIndicator');
  if (!indicator || !activeBtn) return;
  indicator.style.transform = `translateY(${activeBtn.offsetTop}px)`;
  indicator.style.height = `${activeBtn.offsetHeight}px`;
}

function initNavigation() {
  const tabs = document.querySelectorAll('.nav-item');
  const titles = {
    signals: { title: 'Right Now', sub: "What is happening in the athlete's body at this moment." },
    predictions: { title: 'How Tired?', sub: 'How hard the athlete is working, in plain words.' },
    explainability: { title: 'Why?', sub: "What is behind the current reading." },
    personalization: { title: 'About You', sub: 'What we know about this athlete.' },
    agents: { title: 'Coach', sub: 'Advice on what to do next.' },
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');
      positionNavIndicator(tab);
      document.querySelectorAll('.tab-content').forEach((content) => content.classList.remove('active'));
      const activeEl = document.getElementById(`tab-${tab.dataset.tab}`);
      if (activeEl) activeEl.classList.add('active');
      const info = titles[tab.dataset.tab];
      if (info) {
        setText('pageTitle', info.title);
        setText('pageSubtitle', info.sub);
      }
    });
  });

  positionNavIndicator(document.querySelector('.nav-item.active') || tabs[0]);
  window.addEventListener('resize', () => positionNavIndicator(document.querySelector('.nav-item.active')));

  // Charts drawn while a section is closed have no size; fill them when it opens.
  document.getElementById('rawSignalsDetails').addEventListener('toggle', (e) => {
    if (e.target.open && lastPayload) {
      updateRawSignalCharts(lastPayload);
      ['emg', 'imu', 'ppg'].forEach((k) => charts[k] && charts[k].resize());
    }
  });
}

function initPlaybackToggle() {
  const btn = document.getElementById('togglePlayback');
  const pausedBadge = document.getElementById('pausedBadge');
  btn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    setText('playbackBtnText', isPlaying ? 'Pause' : 'Resume');
    // Lucide turns <i> into a static <svg>, so swap in a fresh placeholder to change the icon.
    document.getElementById('playbackIconWrap').innerHTML = `<i data-lucide="${isPlaying ? 'pause' : 'play'}" style="width: 14px; height: 14px;"></i>`;
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

const THEME_STORAGE_KEY = 'athlete-intelligence-theme';

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem(THEME_STORAGE_KEY, theme);
  document.getElementById('themeToggleIconWrap').innerHTML = `<i data-lucide="${theme === 'light' ? 'sun' : 'moon'}" style="width: 14px; height: 14px;"></i>`;
  setText('themeToggleText', theme === 'light' ? 'Light' : 'Dark');
  refreshIcons();
  restyleChartsForTheme();
}

function initThemeToggle() {
  applyTheme(localStorage.getItem(THEME_STORAGE_KEY) === 'light' ? 'light' : 'dark');
  document.getElementById('themeToggle').addEventListener('click', () => {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
  });
}

// ---------------------------------------------------------------------------
// Coach -- real backend only. The advice mirrors the Decision Agent's own
// three outcomes and numbers (backend/agents/decision_agent.py); its exact
// sentence is always shown underneath.
// ---------------------------------------------------------------------------

const PLAIN_ADVICE = {
  CONTINUE_MONITORING: { status: 'Keep going', tone: '', text: "You're in a safe range. Keep going. The coach will keep watching." },
  REST_RECOMMENDED: { status: 'Slow down', tone: 'tone-alert', text: "You're working hard. Lower the load by about 20%, or rest for around a minute." },
  IMMEDIATE_TERMINATION: { status: 'Stop and rest', tone: 'tone-alert', text: 'Stop the exercise now and rest for at least 3 minutes.' },
};

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
  if (feed) feed.innerHTML = '<div class="coach-placeholder">Tap "Check in" to see each step.</div>';
  document.getElementById('recommendationCardSlot').innerHTML = '';
  resetAgentBadges();
  document.getElementById('triggerAgentCheck').disabled = true;
  setText('currentRpeVsThreshold', 'Waiting for live data...');
}

function addCoachBubble({ name, text }) {
  const feed = document.getElementById('agentLogs');
  const placeholder = feed.querySelector('.coach-placeholder');
  if (placeholder) placeholder.remove();
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const bubble = document.createElement('div');
  bubble.className = 'coach-bubble';
  bubble.innerHTML = `
    <div class="coach-body">
      <div class="coach-meta"><span class="coach-agent-name">${name}</span><span class="coach-timestamp">${time}</span></div>
      <div class="coach-text">${text}</div>
    </div>`;
  feed.prepend(bubble);
}

function updateRpeVsThreshold() {
  if (!lastPayload) return;
  if (!estimateReady(lastPayload)) {
    setText('currentRpeVsThreshold', `The coach can give advice once the careful reading starts (after ${warmupSeconds(lastPayload)} seconds of this exercise).`);
    return;
  }
  const current = currentRpeEstimate(lastPayload).value;
  const threshold = parseFloat(document.getElementById('alertThreshold').value);
  const cls = current >= threshold ? 'over-threshold' : 'under-threshold';
  document.getElementById('currentRpeVsThreshold').innerHTML =
    `Right now: <span class="${cls}">${effortWord(current)} (${current.toFixed(1)})</span> · You'll be warned at: ${effortWord(threshold)} (${threshold.toFixed(1)})`;
}

function renderCoachThinking() {
  document.getElementById('recommendationCardSlot').innerHTML =
    '<div class="recommendation-card margin-top"><div class="thinking"><span class="thinking-dots"><span></span><span></span><span></span></span> The coach is looking at the current moment...</div></div>';
}

function renderRecommendationCard(decisionData, effortValue) {
  const advice = PLAIN_ADVICE[decisionData.action_required] || { status: decisionData.action_required, tone: '', text: decisionData.recommendation };
  lastAdvice = { status: advice.status, text: advice.text };
  const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const slot = document.getElementById('recommendationCardSlot');
  slot.innerHTML = `
    <div class="recommendation-card ${advice.tone} margin-top">
      <div class="recommendation-header">
        <span class="advice-status">${advice.status}</span>
        <span class="coach-timestamp">${time}</span>
      </div>
      <p class="advice-text">${advice.text}</p>
      <p class="advice-context">Effort right now: ${effortLabel(effortValue)}</p>
      <details class="advice-original"><summary>The coach's exact words</summary><p>${decisionData.recommendation}</p></details>
    </div>`;
  slot.querySelector('.recommendation-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function badgeForDataAgent(res) {
  if (!res.success) return { text: 'Problem found', tone: 'badge-alert' };
  return res.data && res.data.warnings_count ? { text: 'Warnings', tone: 'badge-moderate' } : { text: 'All good', tone: 'badge-success' };
}

function initAgentLoop() {
  const triggerBtn = document.getElementById('triggerAgentCheck');
  const slider = document.getElementById('alertThreshold');

  slider.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    setText('thresholdVal', `${v.toFixed(1)} (${effortWord(v)})`);
    updateRpeVsThreshold();
  });

  triggerBtn.addEventListener('click', async () => {
    if (!lastPayload) return;
    const threshold = parseFloat(slider.value);
    const effort = currentRpeEstimate(lastPayload).value;
    triggerBtn.disabled = true;
    triggerBtn.textContent = 'Checking...';
    Object.keys(AGENT_BADGE_IDS).forEach((key) => setAgentBadge(key, 'Working...', 'badge-neutral'));
    renderCoachThinking();

    try {
      const response = await fetch(`${API_BASE}/api/agents/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_id: lastPayload.subject_id, step: lastPayload.step, threshold }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }
      const pipeline = (await response.json()).agent_pipeline_results;

      if (pipeline.decision_agent) {
        renderRecommendationCard(pipeline.decision_agent.data, effort);
        updateHero(lastPayload);
        const tone = pipeline.decision_agent.data.action_required === 'CONTINUE_MONITORING' ? 'badge-success'
          : pipeline.decision_agent.data.action_required === 'REST_RECOMMENDED' ? 'badge-moderate' : 'badge-alert';
        setAgentBadge('decision_agent', (PLAIN_ADVICE[pipeline.decision_agent.data.action_required] || {}).status || 'Done', tone);
      }
      if (pipeline.data_agent) {
        addCoachBubble({ name: 'Signal check', text: pipeline.data_agent.narrative });
        const b = badgeForDataAgent(pipeline.data_agent);
        setAgentBadge('data_agent', b.text, b.tone);
      }
      if (pipeline.personalization_agent) {
        addCoachBubble({ name: 'Personal adjustment', text: pipeline.personalization_agent.narrative });
        setAgentBadge('personalization_agent', 'Done', 'badge-success');
      }
      if (pipeline.explainability_agent) {
        addCoachBubble({ name: 'Explainer', text: pipeline.explainability_agent.narrative });
        setAgentBadge('explainability_agent', 'Done', 'badge-success');
      }
      clearError();
    } catch (err) {
      document.getElementById('recommendationCardSlot').innerHTML =
        `<div class="recommendation-card tone-alert margin-top"><p class="advice-text">The coach could not check in: ${err.message}.</p></div>`;
      Object.keys(AGENT_BADGE_IDS).forEach((key) => setAgentBadge(key, 'Unavailable', 'badge-alert'));
    } finally {
      triggerBtn.disabled = false;
      triggerBtn.textContent = 'Check in';
    }
  });

  // Ask a question (answers are grounded in the live window and cite their sources)
  const askBtn = document.getElementById('askDataBtn');
  const askInput = document.getElementById('askDataInput');
  const ask = () => {
    const query = askInput.value.trim();
    if (query) fetchRAGExplanation(query);
  };
  askBtn.addEventListener('click', ask);
  askInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') ask(); });
  document.querySelectorAll('[data-ask]').forEach((chip) => {
    chip.addEventListener('click', () => {
      askInput.value = chip.dataset.ask;
      ask();
    });
  });
}

// The server recomputes every number it cites from this exact window; on any
// failure the box shows the real reason -- never a canned answer.
// The backend answer names model features (slope_sign_changes) and internal tags
// (CALIBRATED_TO_P01); show the everyday label instead, and escape the text first.
function plainRagText(text) {
  let out = String(text)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/\s*\(CALIBRATED_TO_[^)]*\)/g, '');
  Object.keys(FEATURE_TRANSLATIONS).forEach((key) => {
    out = out.replace(new RegExp(`\\b${key}\\b`, 'g'), FEATURE_TRANSLATIONS[key].label.toLowerCase());
  });
  return out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>').replace(/\n/g, '<br>');
}

async function fetchRAGExplanation(query) {
  const contentBox = document.getElementById('ragExplainContent');
  const sourceTagsBox = document.getElementById('ragSourceTags');
  const showPanelError = (message) => {
    contentBox.textContent = message;
    sourceTagsBox.innerHTML = '';
  };

  if (!lastPayload) {
    showPanelError("There's no live data yet. Wait for the athlete's view to start, then ask again.");
    return;
  }
  contentBox.textContent = 'Looking that up...';
  sourceTagsBox.innerHTML = '';

  try {
    const res = await fetch(`${API_BASE}/api/rag/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ subject_id: lastPayload.subject_id, step: lastPayload.step, user_query: query }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      showPanelError(`No answer available: ${body.detail || `HTTP ${res.status}`}`);
      return;
    }
    const data = await res.json();
    contentBox.innerHTML = plainRagText(data.explanation);
    sourceTagsBox.innerHTML = (data.retrieved_sources || []).map((s) => `<span class="chip">${s}</span>`).join(' ');
  } catch (e) {
    showPanelError(`No answer available: the backend could not be reached (${e.message}).`);
  }
}

// ---------------------------------------------------------------------------
// Start-up
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
