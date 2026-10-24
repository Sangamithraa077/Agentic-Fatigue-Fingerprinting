// Athlete Intelligence -- dashboard.
//
// Every figure shown here comes from the FastAPI backend's real endpoints
// (`/api/replay/*`, `/api/agents/analyze`, `/api/rag/explain`, `/api/health`),
// which replay a chosen athlete's actual WSD4FEDSRM recording through this
// project's real preprocessing, models, SHAP explainer and agents. Nothing in
// this file generates or approximates data; if the backend is unavailable the
// page says so instead of showing anything made up.
//
// This file's job is translation: turning model output into plain, precise
// words an athlete or coach can act on ("Hard, 16 out of 20", "Higher than
// usual"), and presenting the AI coach as a verifiable conversation.

// Served by the backend itself (port 8000) -> same origin; otherwise (vite dev, file://) -> local backend.
const API_BASE = window.API_BASE
  ?? (location.protocol.startsWith('http') && location.port === '8000' ? '' : 'http://127.0.0.1:8000');
const RING_CIRCUMFERENCE = 326.7; // 2 * PI * r(52), matches the SVG ring in index.html
const MAX_HISTORY_POINTS = 120;
const CHAT_CONTEXT_TURNS = 6; // earlier messages sent along so follow-up questions keep their context
const WARN_LEVEL = 15;
const STOP_LEVEL = 18;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let isPlaying = true;
let currentEventSource = null;
let lastPayload = null;
let lastAdvice = null; // { status, text, time } from the latest safety check
let alertActive = false;
let chatBusy = false;
const effortHistory = []; // {t, rms} -- muscle effort per window
const ratingHistory = []; // {t, estimate, reported}
const chatHistory = []; // {role: 'user' | 'assistant', content}
const transcript = []; // {question, answer, reading, source} -- for the exported report

const charts = {};
const counterAnimations = new WeakMap();

const $ = (id) => document.getElementById(id);

function refreshIcons() {
  if (window.lucide) window.lucide.createIcons();
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function setText(id, text) {
  const el = $(id);
  if (el) el.textContent = text;
}

// Everything that came from the network is escaped before it touches innerHTML.
function esc(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function capitalize(text) {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function clockTime(withSeconds = false) {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', ...(withSeconds ? { second: '2-digit' } : {}) });
}

// ---------------------------------------------------------------------------
// Plain-language translation
// ---------------------------------------------------------------------------

// The standard Borg 6-20 verbal anchors, which athletes and coaches know.
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
  if (value >= STOP_LEVEL) return { word: 'Critical', tone: 'tone-alert' };
  if (value >= WARN_LEVEL) return { word: 'Fatigued', tone: 'tone-moderate' };
  if (value >= 11) return { word: 'Working', tone: 'tone-good' };
  return { word: 'Fresh', tone: 'tone-good' };
}

function effortLabel(value) {
  return `${effortWord(value)} · ${value.toFixed(0)} of 20`;
}

// z = how many of this athlete's own standard deviations a signal sits from
// their own average (from the backend's per-athlete baseline).
function usualWord(z) {
  if (z === undefined || z === null) return 'Not available';
  if (z >= 1) return 'Well above usual';
  if (z >= 0.4) return 'Above usual';
  if (z <= -1) return 'Well below usual';
  if (z <= -0.4) return 'Below usual';
  return 'Typical for this athlete';
}

// Median frequency of the muscle signal drifts down as a muscle tires.
function fatigueMarkerWord(z) {
  if (z === undefined || z === null) return 'Not available';
  if (z <= -1) return 'Strong';
  if (z <= -0.4) return 'Emerging';
  return 'Not detected';
}

// Plain names and one-line meanings for the 12 real signal features the model uses.
// Mirrored in ml/rag/feature_glossary.py -- keep the two in step.
const FEATURE_TRANSLATIONS = {
  mav: { label: 'Average muscle effort', desc: 'How hard the muscle worked on average.' },
  rms: { label: 'Muscle activity level', desc: "The overall strength of the muscle signal." },
  variance: { label: 'Effort variability', desc: 'How much the effort fluctuated.' },
  waveform_length: { label: 'Signal complexity', desc: 'How busy the muscle signal is.' },
  zero_crossings: { label: 'Signal fluctuation rate', desc: 'How quickly the muscle signal changes direction.' },
  slope_sign_changes: { label: 'Muscle firing pattern', desc: 'How the muscle fibres are firing.' },
  willison_amplitude: { label: 'Contraction bursts', desc: 'How many strong contractions occurred.' },
  integrated_emg: { label: 'Total muscle engagement', desc: 'The total muscle activity in this window.' },
  mean_frequency: { label: 'Muscle signal speed', desc: 'A rhythm that slows down as muscles tire.' },
  median_frequency: { label: 'Muscle fibre tiredness', desc: 'A rhythm that slows down as muscles tire.' },
  peak_frequency: { label: 'Dominant muscle rhythm', desc: 'The strongest rhythm in the muscle signal.' },
  spectral_entropy: { label: 'Signal spread', desc: 'How spread out the muscle signal is.' },
};

function translateFeature(key) {
  return FEATURE_TRANSLATIONS[key] || { label: key.replace(/_/g, ' '), desc: 'A signal feature used by the model.' };
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
    return `Method: this athlete's signals are compared with ${borg.trained_on_athletes} other athletes. The estimate is typically off by about ${borg.typical_error_borg} points on the 6–20 scale, so treat it as a guide, not a measurement.`;
  }
  if (estimateReady(payload)) return 'Method: a quick estimate from the fatigue detector. No comparison data is available for this athlete.';
  return `Calibrating: the reliable estimate starts after ${warmupSeconds(payload)} seconds of this exercise. Until then the fatigue detector's rough guess is ${effortLabel(currentRpeEstimate(payload).value).toLowerCase()}, which should not be relied on.`;
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
  const el = $('connectionStatus');
  el.classList.remove('status-error', 'status-loading');
  if (state === 'error') el.classList.add('status-error');
  if (state === 'loading') el.classList.add('status-loading');
  setText('connectionStatusText', text);
}

function showError(message) {
  const banner = $('errorBanner');
  banner.textContent = message;
  banner.hidden = false;
}

function clearError() {
  const banner = $('errorBanner');
  banner.hidden = true;
  banner.textContent = '';
}

// Which engine writes the coach's answers (reported by the backend, never assumed).
async function loadEngineStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const h = await res.json();
    const names = {
      anthropic: `AI engine: Claude (${h.llm_model})`,
      ollama: `AI engine: local model (${h.llm_model})`,
      custom: 'AI engine: custom model',
      template: 'AI engine: rule-based answers',
    };
    setText('aiEngineText', names[h.llm_backend] || 'AI engine: unknown');
    $('aiEngineStatus').title = h.llm_backend === 'template'
      ? 'No language model is connected. Answers are built directly from the measurements. Set ANTHROPIC_API_KEY on the backend to enable AI-written answers.'
      : 'Answers are written by this model, then every figure in them is checked against the live data.';
  } catch {
    setText('aiEngineText', 'AI engine: unavailable');
  }
}

// ---------------------------------------------------------------------------
// Charts (empty until real data arrives)
// ---------------------------------------------------------------------------

function getChartPalette() {
  const isLight = document.documentElement.getAttribute('data-theme') === 'light';
  return isLight
    ? {
        text: '#3b475f', muted: '#66738f', grid: 'rgba(15,23,42,0.07)',
        tooltipBg: 'rgba(255,255,255,0.98)', tooltipBorder: 'rgba(15,23,42,0.15)', tooltipTitle: '#0f172a', tooltipBody: '#3b475f',
        cyan: '#0b7fa0', blue: '#3b5bdb', purple: '#7c3aed', amber: '#a35a00', crimson: '#c42a44',
        good: 'rgba(19,122,85,0.07)', warn: 'rgba(163,90,0,0.10)', bad: 'rgba(196,42,68,0.11)', line: 'rgba(15,23,42,0.28)',
        cyanFillStrong: 'rgba(11,127,160,0.22)', cyanFillWeak: 'rgba(11,127,160,0.0)',
      }
    : {
        text: '#b6c2db', muted: '#8392b0', grid: 'rgba(148,170,214,0.08)',
        tooltipBg: 'rgba(11,17,32,0.96)', tooltipBorder: 'rgba(148,170,214,0.3)', tooltipTitle: '#eef2fa', tooltipBody: '#b6c2db',
        cyan: '#4cc9f0', blue: '#6d8cff', purple: '#a78bfa', amber: '#fbbf24', crimson: '#fb7185',
        good: 'rgba(52,211,153,0.07)', warn: 'rgba(251,191,36,0.09)', bad: 'rgba(251,113,133,0.11)', line: 'rgba(238,242,250,0.3)',
        cyanFillStrong: 'rgba(76,201,240,0.3)', cyanFillWeak: 'rgba(76,201,240,0.0)',
      };
}

// Shades the Borg zones (light / moderate / hard / maximal) behind the line and
// draws the warning (15) and stop (18) levels.
const borgZonesPlugin = {
  id: 'borgZones',
  beforeDatasetsDraw(chart) {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.y) return;
    const p = getChartPalette();
    const y = (v) => scales.y.getPixelForValue(v);
    const band = (from, to, color) => {
      ctx.fillStyle = color;
      ctx.fillRect(chartArea.left, y(to), chartArea.right - chartArea.left, y(from) - y(to));
    };
    ctx.save();
    band(6, 11, p.good); band(11, WARN_LEVEL, 'transparent'); band(WARN_LEVEL, STOP_LEVEL, p.warn); band(STOP_LEVEL, 20, p.bad);
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = p.line;
    ctx.lineWidth = 1;
    ctx.fillStyle = p.muted;
    ctx.font = '11px Inter, sans-serif';
    [[WARN_LEVEL, 'Warning'], [STOP_LEVEL, 'Stop']].forEach(([level, label]) => {
      ctx.beginPath();
      ctx.moveTo(chartArea.left, y(level));
      ctx.lineTo(chartArea.right, y(level));
      ctx.stroke();
      ctx.fillText(label, chartArea.right - 44, y(level) - 4);
    });
    ctx.restore();
  },
};

function initCharts() {
  const p = getChartPalette();
  const font = { family: 'Inter', size: 11 };
  const common = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 250 },
    interaction: { mode: 'index', intersect: false },
    plugins: {
      legend: { labels: { color: p.text, font, usePointStyle: true, boxWidth: 7 } },
      tooltip: { backgroundColor: p.tooltipBg, borderColor: p.tooltipBorder, borderWidth: 1, titleColor: p.tooltipTitle, bodyColor: p.tooltipBody, padding: 10, cornerRadius: 8 },
    },
    scales: {
      x: { grid: { color: p.grid }, ticks: { color: p.muted, font, maxTicksLimit: 8 } },
      y: { grid: { color: p.grid }, ticks: { color: p.muted, font } },
    },
  };
  const fill = (ctx) => {
    const g = ctx.createLinearGradient(0, 0, 0, 240);
    g.addColorStop(0, p.cyanFillStrong);
    g.addColorStop(1, p.cyanFillWeak);
    return g;
  };
  const axisTitle = (text) => ({ display: true, text, color: p.muted, font });
  const withYTitle = (text) => ({ ...common, scales: { ...common.scales, y: { ...common.scales.y, title: axisTitle(text) } } });

  charts.effortTrend = new Chart($('effortTrendChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [{ label: 'Muscle effort', data: [], borderColor: p.cyan, borderWidth: 2.25, pointRadius: 0, tension: 0.35, fill: true, backgroundColor: (c) => fill(c.chart.ctx) }] },
    options: {
      ...common,
      plugins: { ...common.plugins, legend: { display: false }, tooltip: { ...common.plugins.tooltip, enabled: false } },
      scales: { x: { ...common.scales.x, title: axisTitle('Seconds into the exercise') }, y: { ...common.scales.y, min: 0, ticks: { display: false }, title: axisTitle('Relative effort') } },
    },
  });

  charts.progression = new Chart($('fatigueProgressionChart').getContext('2d'), {
    type: 'line',
    plugins: [borgZonesPlugin],
    data: {
      labels: [],
      datasets: [
        { label: 'Estimated effort', data: [], borderColor: p.cyan, borderWidth: 2.5, pointRadius: 0, tension: 0.3 },
        { label: "Athlete's rating", data: [], borderColor: p.amber, borderDash: [6, 4], borderWidth: 2, pointRadius: 3, spanGaps: true },
      ],
    },
    options: {
      ...common,
      scales: {
        x: { ...common.scales.x, title: axisTitle('Seconds into the exercise') },
        y: { ...common.scales.y, min: 6, max: 20, ticks: { ...common.scales.y.ticks, stepSize: 2 }, title: axisTitle('Borg effort (6–20)') },
      },
    },
  });

  // Technical raw-signal charts (shown only inside the collapsed "Raw sensor signals")
  const line = (label, color) => ({ label, data: [], borderColor: color, borderWidth: 1.5, pointRadius: 0, tension: 0.2 });
  charts.emg = new Chart($('emgChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [line('Front shoulder', p.cyan), line('Back shoulder', p.purple), line('Upper back', p.amber)] },
    options: withYTitle('Muscle signal (mV)'),
  });
  charts.imu = new Chart($('imuChart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [line('Movement, axis 1', p.crimson), line('Movement, axis 2', p.blue), line('Movement, axis 3', p.amber)] },
    options: withYTitle('Arm movement (m/s²)'),
  });
  charts.ppg = new Chart($('ppgChart').getContext('2d'), {
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
  const select = $('subjectSelect');
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
    } else {
      setConnectionStatus('error', 'No recordings found');
      showError('The backend is running but found no athlete recordings. Place the WSD4FEDSRM dataset under datasets/raw/ and restart it.');
      select.innerHTML = '<option value="">No athletes</option>';
    }
  } catch (err) {
    setConnectionStatus('error', 'Backend unreachable');
    showError(`The athlete list could not be loaded (${err.message}). Start the backend with: uvicorn backend.api.app:app --reload`);
    select.innerHTML = '<option value="">Backend unavailable</option>';
  }
}

function resetSessionState() {
  lastPayload = null;
  lastAdvice = null;
  alertActive = false;
  effortHistory.length = 0;
  ratingHistory.length = 0;
  chatHistory.length = 0;
  transcript.length = 0;
  refreshHistoryCharts();
  resetCoachFeed();
  resetChat();
  $('alertBanner').hidden = true;
  $('exportReport').disabled = true;
  setText('heroRecommendation', 'Run a safety check in the AI coach tab to get a recommendation.');
  setText('heroRecommendationSub', '');
}

function startReplayStream(subjectId) {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }
  resetSessionState();
  setConnectionStatus('loading', `Loading athlete ${subjectId}… first load can take a few minutes`);

  const source = new EventSource(`${API_BASE}/api/replay/stream/${encodeURIComponent(subjectId)}?speed_ms=600`);
  currentEventSource = source;

  source.onopen = () => {
    setConnectionStatus('connected', `Live · athlete ${subjectId}`);
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
  updateLiveMonitor(payload);
  updateEffortAnalysis(payload);
  updateDrivers(payload);
  updateProfile(payload);
  $('triggerAgentCheck').disabled = !estimateReady(payload);
  $('exportReport').disabled = false;
  updateRpeVsThreshold();
  updateAlert(payload);
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

function exerciseName(payload) {
  return capitalize(payload.direction.replace(/_/g, ' '));
}

function updateHero(payload) {
  const value = currentRpeEstimate(payload).value;
  const pill = $('heroStatusPill');
  const chip = $('heroErrorChip');
  const borg = payload.prediction.borg_regression || {};

  setText('heroAthleteId', payload.subject_id);
  setText('heroActivity', ` · ${exerciseName(payload)} · ${payload.intensity_band}% MVC`);

  if (estimateReady(payload)) {
    const status = effortStatus(value);
    animateCounter($('heroScoreValue'), value, { decimals: 0 });
    setRingProgress($('heroRingProgress'), ((value - 6) / 14) * 100);
    $('heroStatusText').textContent = status.word;
    pill.className = `status-pill ${status.tone}`;
    setText('heroEffortSub', `Perceived effort is ${effortWord(value).toLowerCase()} (${value.toFixed(1)} of 20).`);
    chip.hidden = !(borg.typical_error_borg != null);
    if (!chip.hidden) chip.textContent = `± ${borg.typical_error_borg} typical error`;
  } else {
    const valueEl = $('heroScoreValue');
    valueEl.dataset.rawValue = '0';
    valueEl.textContent = '--';
    setRingProgress($('heroRingProgress'), 0);
    $('heroStatusText').textContent = 'Calibrating';
    pill.className = 'status-pill tone-neutral';
    setText('heroEffortSub', `Reliable readings begin after ${warmupSeconds(payload)} seconds of this exercise.`);
    chip.hidden = true;
  }

  if (lastAdvice) {
    setText('heroRecommendation', lastAdvice.text);
    setText('heroRecommendationSub', `${lastAdvice.status} · safety check at ${lastAdvice.time}`);
  } else if (estimateReady(payload)) {
    setText('heroRecommendation', 'Run a safety check in the AI coach tab to get a recommendation.');
    setText('heroRecommendationSub', '');
  } else {
    setText('heroRecommendation', 'Recommendations become available once the reliable estimate starts.');
    setText('heroRecommendationSub', '');
  }
}

// Surfaces the moment effort crosses the warning level, once per crossing.
function updateAlert(payload) {
  const banner = $('alertBanner');
  if (!estimateReady(payload)) return;
  const value = currentRpeEstimate(payload).value;
  const threshold = parseFloat($('alertThreshold').value);
  if (value >= threshold) {
    const critical = value >= STOP_LEVEL;
    banner.classList.toggle('is-critical', critical);
    banner.innerHTML = `<span><strong>${critical ? 'Critical effort' : 'Effort threshold reached'}:</strong> ${esc(effortWord(value))} (${value.toFixed(1)} of 20). Run a safety check for a recommendation.</span><button type="button" id="alertAction">Run safety check</button>`;
    banner.hidden = false;
    $('alertAction').addEventListener('click', () => { switchTab('agents'); $('triggerAgentCheck').click(); });
    alertActive = true;
  } else if (alertActive) {
    banner.hidden = true;
    alertActive = false;
  }
}

function updateLiveMonitor(payload) {
  const zscores = payload.personalization.feature_baseline_zscores || {};
  setText('activityText', `${exerciseName(payload)} · ${payload.intensity_band}% MVC`);
  setText('tileEffortValue', usualWord(zscores.rms));
  setText('tileFatigueValue', fatigueMarkerWord(zscores.median_frequency));
  const reported = payload.ground_truth.rpe_nearest;
  setText('tileReportedValue', reported === null ? 'No rating nearby' : effortLabel(reported));

  const seconds = Math.round(payload.elapsed_seconds);
  setText('tileProgressValue', `${seconds} s`);
  if (payload.total_steps) $('tileProgressBar').style.width = `${clamp((payload.step / payload.total_steps) * 100, 0, 100)}%`;
  updateRawSignalCharts(payload);
}

function updateRawSignalCharts(payload) {
  if (!$('rawSignalsDetails').open) return; // only draw what is visible
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

function updateEffortAnalysis(payload) {
  const pred = payload.prediction;
  const value = currentRpeEstimate(payload).value;
  const reported = payload.ground_truth.rpe_nearest;
  const ready = estimateReady(payload);

  setText('effortHeadline', ready ? `${effortWord(value)} · ${value.toFixed(1)} of 20 on the Borg scale` : `Calibrating. Reliable readings begin after ${warmupSeconds(payload)} seconds.`);
  setText('estimatedRpeText', ready ? effortLabel(value) : 'Not ready yet');
  setText('groundTruthRpeText', reported === null ? 'No rating nearby' : effortLabel(reported));
  setText('fatigueRiskText', `${Math.round(pred.probability_fatigued * 100)}%`);
  setText('estimatedRpeSource', describeEstimateMethod(payload));

  const toPercent = (v) => `${clamp(((v - 6) / 14) * 100, 0, 100)}%`;
  const estimateMarker = $('meterEstimate');
  estimateMarker.hidden = !ready;
  if (ready) estimateMarker.style.left = toPercent(value);
  const reportedMarker = $('meterReported');
  reportedMarker.hidden = reported === null;
  if (reported !== null) reportedMarker.style.left = toPercent(reported);
}

function updateDrivers(payload) {
  const local = payload.explainability.local_top_features.slice().sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)).slice(0, 3);
  const list = $('driverList');
  if (!local.length) {
    list.innerHTML = '<p class="empty">Waiting for live data…</p>';
    return;
  }
  const maxAbs = Math.max(...local.map((f) => Math.abs(f.shap)), 1e-9);
  list.innerHTML = local.map((f, i) => {
    const t = translateFeature(f.feature);
    const up = f.shap >= 0;
    return `
      <div class="driver">
        <div class="driver-rank">${i + 1}</div>
        <div>
          <div class="driver-name">${esc(t.label)}</div>
          <div class="driver-desc">${esc(t.desc)}</div>
          <div class="driver-bar" aria-hidden="true"><span class="${up ? 'up' : 'down'}" style="width:${(Math.abs(f.shap) / maxAbs) * 100}%"></span></div>
        </div>
        <div class="driver-side">
          <span class="chip ${up ? 'chip-up' : 'chip-down'}">${up ? 'Raising the reading' : 'Lowering the reading'}</span>
          <button type="button" class="link-btn" data-explain="${esc(t.label.toLowerCase())}">Ask the coach</button>
        </div>
      </div>`;
  }).join('');
  setText('driversLede', `Mainly ${translateFeature(local[0].feature).label.toLowerCase()}, which is ${local[0].shap >= 0 ? 'raising' : 'lowering'} the current reading.`);

  setText('shapTechDetails', local.map((f) => `${f.feature}: ${f.shap >= 0 ? '+' : ''}${f.shap.toFixed(4)}  (value ${f.raw_value.toFixed(4)})`).join('\n'));
  setText('shapGlobalDetails', payload.explainability.global_top_features.map((f) => `${f.feature}: ${f.importance.toFixed(4)}`).join('\n'));
}

function updateProfile(payload) {
  const profile = payload.personalization.profile;
  setText('profileSubjectId', `Athlete ${payload.subject_id}`);
  setText('profileAvatar', payload.subject_id.replace(/[^0-9A-Za-z]/g, '').slice(0, 3).toUpperCase());

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
    setText('historyText', `Across ${h.n_trials} earlier exercises, this athlete rated effort between ${h.min_rpe.toFixed(0)} (${effortWord(h.min_rpe).toLowerCase()}) and ${h.max_rpe.toFixed(0)} (${effortWord(h.max_rpe).toLowerCase()}), averaging ${h.mean_rpe.toFixed(1)} of 20.`);
  } else {
    setText('historyText', 'No earlier effort ratings are on record for this athlete.');
  }

  setText('genericRiskText', `${Math.round(payload.personalization.generic_probability * 100)}%`);
  setText('personalRiskText', `${Math.round(payload.personalization.personalized_probability * 100)}%`);
}

// ---------------------------------------------------------------------------
// Navigation, playback, theme, shortcuts
// ---------------------------------------------------------------------------

const TAB_TITLES = {
  signals: { title: 'Live monitor', sub: 'Real-time muscle response for the selected athlete.' },
  predictions: { title: 'Effort analysis', sub: 'Perceived effort on the Borg scale, with the method and its typical error.' },
  explainability: { title: 'Drivers', sub: 'Which signals are moving the current reading, and in which direction.' },
  personalization: { title: 'Athlete profile', sub: 'The background used to compare this athlete with their own usual signals.' },
  agents: { title: 'AI coach', sub: 'Ask questions, get recommendations, and see the evidence behind them.' },
  lab: { title: 'Session lab', sub: 'Replay a recorded exercise: estimate versus the athlete\'s own rating, personalisation, drivers and explanation.' },
};
const TAB_ORDER = Object.keys(TAB_TITLES);

function switchTab(name) {
  if (!TAB_TITLES[name]) return;
  document.querySelectorAll('.nav-item').forEach((btn) => {
    const active = btn.dataset.tab === name;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-selected', String(active));
  });
  document.querySelectorAll('.tab-panel').forEach((panel) => panel.classList.toggle('active', panel.id === `tab-${name}`));
  setText('pageTitle', TAB_TITLES[name].title);
  setText('pageSubtitle', TAB_TITLES[name].sub);
  document.body.dataset.tab = name; // lets the stylesheet hide the live-replay strip on the Session lab
  // Charts drawn while hidden have no size; refit them when their tab opens.
  Object.values(charts).forEach((c) => c && c.resize());
  window.dispatchEvent(new CustomEvent('tabchange', { detail: name }));
}

function initNavigation() {
  document.querySelectorAll('.nav-item').forEach((tab) => tab.addEventListener('click', () => switchTab(tab.dataset.tab)));

  // Charts drawn while a section is closed have no size; fill them when it opens.
  $('rawSignalsDetails').addEventListener('toggle', (e) => {
    if (e.target.open && lastPayload) {
      updateRawSignalCharts(lastPayload);
      ['emg', 'imu', 'ppg'].forEach((k) => charts[k] && charts[k].resize());
    }
  });

  // "Ask the coach" next to a driver
  $('driverList').addEventListener('click', (e) => {
    const btn = e.target.closest('[data-explain]');
    if (!btn) return;
    switchTab('agents');
    askCoach(`What does ${btn.dataset.explain} mean for this reading?`);
  });

  document.addEventListener('keydown', (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName || '');
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '/' && !typing) {
      e.preventDefault();
      switchTab('agents');
      $('askDataInput').focus();
    } else if (!typing && /^[1-6]$/.test(e.key)) {
      switchTab(TAB_ORDER[Number(e.key) - 1]);
    } else if (!typing && e.key === ' ' && document.body.dataset.tab === 'lab') {
      // the Session lab handles its own play / pause (lab.js)
    } else if (!typing && e.key === ' ' && document.activeElement?.tagName !== 'BUTTON') {
      e.preventDefault();
      $('togglePlayback').click();
    }
  });
}

function initPlaybackToggle() {
  const btn = $('togglePlayback');
  btn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    setText('playbackBtnText', isPlaying ? 'Pause' : 'Resume');
    // Lucide turns <i> into a static <svg>, so swap in a fresh placeholder to change the icon.
    $('playbackIconWrap').innerHTML = `<i data-lucide="${isPlaying ? 'pause' : 'play'}"></i>`;
    btn.classList.toggle('is-paused', !isPlaying);
    $('pausedBadge').hidden = isPlaying;
    refreshIcons();
  });
}

function initSubjectSelector() {
  $('subjectSelect').addEventListener('change', (e) => {
    if (e.target.value) startReplayStream(e.target.value);
  });
}

const THEME_STORAGE_KEY = 'athlete-intelligence-theme';

function readStoredTheme() {
  try { return localStorage.getItem(THEME_STORAGE_KEY); } catch { return null; }
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  try { localStorage.setItem(THEME_STORAGE_KEY, theme); } catch { /* private mode: the choice just isn't remembered */ }
  $('themeToggleIconWrap').innerHTML = `<i data-lucide="${theme === 'light' ? 'sun' : 'moon'}"></i>`;
  refreshIcons();
  restyleChartsForTheme();
}

function initThemeToggle() {
  const stored = readStoredTheme();
  const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
  applyTheme(stored ? (stored === 'light' ? 'light' : 'dark') : (prefersLight ? 'light' : 'dark'));
  $('themeToggle').addEventListener('click', () => {
    applyTheme(document.documentElement.getAttribute('data-theme') === 'light' ? 'dark' : 'light');
  });
}

// ---------------------------------------------------------------------------
// Safety check -- the four-agent analysis. The advice mirrors the Decision
// Agent's own three outcomes and numbers (backend/agents/decision_agent.py);
// its exact sentence is always available underneath.
// ---------------------------------------------------------------------------

const PLAIN_ADVICE = {
  CONTINUE_MONITORING: { status: 'Keep going', tone: '', stepTone: 'ok', text: 'Effort is in a safe range. Keep going; the coach will keep watching.' },
  REST_RECOMMENDED: { status: 'Slow down', tone: 'tone-moderate', stepTone: 'warn', text: 'Effort is high. Lower the load by about 20%, or rest for around a minute.' },
  IMMEDIATE_TERMINATION: { status: 'Stop and rest', tone: 'tone-alert', stepTone: 'bad', text: 'Stop the exercise now and rest for at least 3 minutes.' },
};

const AGENT_KEYS = ['data_agent', 'personalization_agent', 'explainability_agent', 'decision_agent'];
const AGENT_BADGE_IDS = {
  data_agent: 'dataAgentBadge',
  personalization_agent: 'personalizationAgentBadge',
  explainability_agent: 'explainabilityAgentBadge',
  decision_agent: 'decisionAgentBadge',
};

// tone: 'idle' | 'working' | 'ok' | 'warn' | 'bad'
function setAgentStep(agentKey, text, tone) {
  const badge = $(AGENT_BADGE_IDS[agentKey]);
  if (!badge) return;
  badge.textContent = text;
  const li = badge.closest('li');
  li.className = tone === 'idle' ? '' : `is-${tone}`;
}

function resetCoachFeed() {
  $('agentLogs').innerHTML = '<div class="log-empty">Run a safety check to see each step.</div>';
  $('recommendationCardSlot').innerHTML = '';
  AGENT_KEYS.forEach((key) => setAgentStep(key, 'Ready', 'idle'));
  $('triggerAgentCheck').disabled = true;
  setText('currentRpeVsThreshold', 'Waiting for live data…');
}

function addLogEntry(name, text) {
  const feed = $('agentLogs');
  const empty = feed.querySelector('.log-empty');
  if (empty) empty.remove();
  const item = document.createElement('div');
  item.className = 'log-item';
  item.innerHTML = `<header><strong>${esc(name)}</strong><time>${esc(clockTime(true))}</time></header><p>${esc(text)}</p>`;
  feed.prepend(item);
}

function updateRpeVsThreshold() {
  if (!lastPayload) return;
  if (!estimateReady(lastPayload)) {
    setText('currentRpeVsThreshold', `Available once the reliable estimate starts (after ${warmupSeconds(lastPayload)} seconds of this exercise).`);
    return;
  }
  const current = currentRpeEstimate(lastPayload).value;
  const threshold = parseFloat($('alertThreshold').value);
  const cls = current >= threshold ? 'over-threshold' : 'under-threshold';
  $('currentRpeVsThreshold').innerHTML =
    `Now: <span class="${cls}">${esc(effortWord(current))} (${current.toFixed(1)})</span> · warning at ${esc(effortWord(threshold).toLowerCase())} (${threshold.toFixed(1)})`;
}

function renderCoachThinking() {
  $('recommendationCardSlot').innerHTML = '<div class="reco"><div class="reco-text"><span class="typing"><span></span><span></span><span></span></span> Analysing the current moment…</div></div>';
}

function renderRecommendationCard(decisionData, effortValue) {
  const advice = PLAIN_ADVICE[decisionData.action_required] || { status: decisionData.action_required, tone: '', text: decisionData.recommendation };
  const time = clockTime();
  lastAdvice = { status: advice.status, text: advice.text, time };
  $('recommendationCardSlot').innerHTML = `
    <div class="reco ${advice.tone}">
      <div class="reco-head"><span class="reco-status">${esc(advice.status)}</span><span class="reco-time">${esc(time)}</span></div>
      <p class="reco-text">${esc(advice.text)}</p>
      <p class="reco-context">Effort at the time of the check: ${esc(effortLabel(effortValue))}</p>
      <details><summary>Exact wording from the decision step</summary><p>${esc(decisionData.recommendation)}</p></details>
    </div>`;
}

function dataStepState(res) {
  if (!res.success) return { text: 'Problem found', tone: 'bad' };
  return res.data && res.data.warnings_count ? { text: 'Warnings', tone: 'warn' } : { text: 'All clear', tone: 'ok' };
}

function initSafetyCheck() {
  const triggerBtn = $('triggerAgentCheck');
  const slider = $('alertThreshold');
  const label = triggerBtn.querySelector('span');

  slider.addEventListener('input', (e) => {
    const v = parseFloat(e.target.value);
    setText('thresholdVal', `${v.toFixed(1)} · ${effortWord(v)}`);
    updateRpeVsThreshold();
    if (lastPayload) updateAlert(lastPayload);
  });

  triggerBtn.addEventListener('click', async () => {
    if (!lastPayload) return;
    const threshold = parseFloat(slider.value);
    const effort = currentRpeEstimate(lastPayload).value;
    const { subject_id: subjectId, step } = lastPayload;
    triggerBtn.disabled = true;
    label.textContent = 'Checking…';
    AGENT_KEYS.forEach((key) => setAgentStep(key, 'Working…', 'working'));
    renderCoachThinking();

    try {
      const response = await fetch(`${API_BASE}/api/agents/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ subject_id: subjectId, step, threshold }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || `HTTP ${response.status}`);
      }
      const pipeline = (await response.json()).agent_pipeline_results;

      if (pipeline.data_agent) {
        addLogEntry('Signal quality', pipeline.data_agent.narrative);
        const s = dataStepState(pipeline.data_agent);
        setAgentStep('data_agent', s.text, s.tone);
      }
      if (pipeline.personalization_agent) {
        addLogEntry('Personal adjustment', pipeline.personalization_agent.narrative);
        setAgentStep('personalization_agent', 'Done', 'ok');
      }
      if (pipeline.explainability_agent) {
        addLogEntry('Reasons', pipeline.explainability_agent.narrative);
        setAgentStep('explainability_agent', 'Done', 'ok');
      }
      if (pipeline.decision_agent) {
        renderRecommendationCard(pipeline.decision_agent.data, effort);
        const adv = PLAIN_ADVICE[pipeline.decision_agent.data.action_required] || {};
        setAgentStep('decision_agent', adv.status || 'Done', adv.stepTone || 'ok');
        if (lastPayload) updateHero(lastPayload);
        $('alertBanner').hidden = true;
        alertActive = false;
      }
      clearError();
    } catch (err) {
      $('recommendationCardSlot').innerHTML =
        `<div class="reco tone-alert"><p class="reco-text">The safety check could not run: ${esc(err.message)}.</p></div>`;
      AGENT_KEYS.forEach((key) => setAgentStep(key, 'Unavailable', 'bad'));
    } finally {
      triggerBtn.disabled = !(lastPayload && estimateReady(lastPayload));
      label.textContent = 'Run safety check';
    }
  });
}

// ---------------------------------------------------------------------------
// AI coach chat. The server recomputes every number it cites from the exact
// window we send, verifies the model's figures against it, and tells us whether
// the answer was AI-written or built by rule -- we show that, plus the sources.
// ---------------------------------------------------------------------------

const FALLBACK_REASONS = {
  out_of_scope: 'That question is outside what the sensors measure, so the coach declined to guess.',
  no_llm_configured: 'No language model is connected, so this answer is built directly from the measurements.',
  ungrounded_numbers: 'The AI draft quoted a figure that is not in the data, so a verified summary is shown instead.',
  prohibited_term: 'The AI draft used medical wording, which this system does not allow, so a verified summary is shown instead.',
  llm_error: 'The language model did not respond, so a verified summary is shown instead.',
  llm_empty: 'The language model returned nothing, so a verified summary is shown instead.',
};

// Escape first, then add the little formatting the coach uses: **bold**, line breaks and
// [SOURCE-ID] citations that jump to the matching source card.
function renderRich(text, uid) {
  return esc(text)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\[([A-Z][A-Z0-9-]{3,})\]/g, (_, id) => `<button type="button" class="cite" data-cite="${uid}-${id}" title="Show source ${id}">${id}</button>`)
    .replace(/\n/g, '<br>');
}

function resetChat() {
  const thread = $('chatThread');
  thread.innerHTML = '';
  thread.appendChild($('chatEmpty') || buildChatEmpty());
  $('clearChat').hidden = true;
  refreshIcons();
}

function buildChatEmpty() {
  const div = document.createElement('div');
  div.className = 'chat-empty';
  div.id = 'chatEmpty';
  div.innerHTML = '<i data-lucide="message-circle"></i><strong>Start with a question</strong><span>Try a quick action above, or type your own below.</span>';
  return div;
}

function scrollChat() {
  const thread = $('chatThread');
  thread.scrollTop = thread.scrollHeight;
}

function addUserMessage(text) {
  const empty = $('chatEmpty');
  if (empty) empty.remove();
  $('clearChat').hidden = false;
  const el = document.createElement('div');
  el.className = 'msg msg-user';
  el.innerHTML = `<div class="msg-bubble">${esc(text)}</div>`;
  $('chatThread').appendChild(el);
  scrollChat();
}

function addTypingMessage() {
  const el = document.createElement('div');
  el.className = 'msg msg-coach';
  el.innerHTML = '<div class="msg-bubble"><span class="typing" role="status" aria-label="The coach is typing"><span></span><span></span><span></span></span></div>';
  $('chatThread').appendChild(el);
  scrollChat();
  return el;
}

let messageCounter = 0;

function fillCoachMessage(el, data, reading) {
  const uid = `m${++messageCounter}`;
  const answer = data.answer || data.explanation || '';
  const isAi = data.source === 'llm';
  const reason = FALLBACK_REASONS[data.fallback_reason] || FALLBACK_REASONS.no_llm_configured;
  const cited = new Set([...answer.matchAll(/\[([A-Z][A-Z0-9-]{3,})\]/g)].map((m) => m[1]));
  const sources = (data.source_details || []).slice().sort((a, b) => Number(cited.has(b.chunk_id)) - Number(cited.has(a.chunk_id)));

  el.innerHTML = `
    <div class="msg-bubble">${renderRich(answer, uid)}</div>
    <div class="msg-meta">
      <span class="trust ${isAi ? 'trust-ai' : 'trust-rule'}" title="${esc(isAi ? 'Written by the language model; every figure in it was checked against the live data.' : reason)}">
        <i data-lucide="${isAi ? 'badge-check' : 'list-checks'}"></i>${isAi ? 'AI-written · figures verified' : 'Rule-based · from live data'}
      </span>
      <span>Reading at ${esc(Math.round(reading.elapsed))} s</span>
      <button type="button" class="icon-btn" data-copy title="Copy answer"><i data-lucide="copy"></i>Copy</button>
    </div>
    ${sources.length ? `<div class="sources" aria-label="Sources">${sources.map((s) => `
      <details class="source" data-src="${uid}-${esc(s.chunk_id)}">
        <summary><span class="src-id">${esc(s.chunk_id)}</span><span>${esc(s.title)}</span>${cited.has(s.chunk_id) ? '<span class="chip chip-quiet">Cited</span>' : ''}</summary>
        <div class="src-body">${esc(s.snippet)}<em>Source: ${esc(s.source)}</em></div>
      </details>`).join('')}</div>` : ''}
    ${(data.follow_ups || []).length ? `<div class="follow-ups">${data.follow_ups.map((q) => `<button type="button" data-ask="${esc(q)}">${esc(q)}</button>`).join('')}</div>` : ''}`;
  el.dataset.answer = answer;
  refreshIcons();
  scrollChat();
}

function fillCoachError(el, message) {
  el.classList.add('is-error');
  el.innerHTML = `<div class="msg-bubble">${esc(message)}</div>`;
  scrollChat();
}

async function askCoach(query) {
  query = (query || '').trim();
  if (!query || chatBusy) return;
  if (!lastPayload) {
    addUserMessage(query);
    fillCoachError(addTypingMessage(), 'There is no live data yet. Wait for the athlete view to start, then ask again.');
    return;
  }

  // Pin the question to the exact window on screen now; the stream keeps moving.
  const { subject_id: subjectId, step, elapsed_seconds: elapsed } = lastPayload;
  chatBusy = true;
  $('askDataBtn').disabled = true;
  document.querySelectorAll('.follow-ups').forEach((f) => f.remove()); // only the latest answer offers follow-ups
  addUserMessage(query);
  const pending = addTypingMessage();

  try {
    const res = await fetch(`${API_BASE}/api/rag/explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        subject_id: subjectId, step, user_query: query,
        history: chatHistory.slice(-CHAT_CONTEXT_TURNS),
      }),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      const detail = Array.isArray(body.detail) ? 'the request was not accepted' : body.detail;
      throw new Error(detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    fillCoachMessage(pending, data, { elapsed });
    chatHistory.push({ role: 'user', content: query }, { role: 'assistant', content: data.answer || data.explanation });
    transcript.push({ question: query, answer: data.answer || data.explanation, reading: Math.round(elapsed), source: data.source });
  } catch (err) {
    fillCoachError(pending, `No answer available: ${err.message}.`);
  } finally {
    chatBusy = false;
    $('askDataBtn').disabled = false;
  }
}

function initCoachChat() {
  const input = $('askDataInput');
  $('askForm').addEventListener('submit', (e) => {
    e.preventDefault();
    const q = input.value;
    input.value = '';
    askCoach(q);
  });

  // Quick actions and follow-up chips share one handler.
  document.addEventListener('click', (e) => {
    const askBtn = e.target.closest('[data-ask]');
    if (askBtn) { askCoach(askBtn.dataset.ask); return; }

    const cite = e.target.closest('[data-cite]');
    if (cite) {
      const card = document.querySelector(`details.source[data-src="${CSS.escape(cite.dataset.cite)}"]`);
      if (card) {
        card.open = true;
        card.classList.add('flash');
        card.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        setTimeout(() => card.classList.remove('flash'), 1400);
      }
      return;
    }

    const copy = e.target.closest('[data-copy]');
    if (copy) {
      const text = copy.closest('.msg')?.dataset.answer || '';
      navigator.clipboard?.writeText(text).then(() => {
        copy.innerHTML = '<i data-lucide="check"></i>Copied';
        refreshIcons();
        setTimeout(() => { copy.innerHTML = '<i data-lucide="copy"></i>Copy'; refreshIcons(); }, 1400);
      }).catch(() => {});
    }
  });

  $('clearChat').addEventListener('click', () => {
    chatHistory.length = 0;
    resetChat();
  });
}

// ---------------------------------------------------------------------------
// Export -- a plain-text session report built only from what this page received.
// ---------------------------------------------------------------------------

function buildReport() {
  const p = lastPayload;
  const est = estimateReady(p) ? currentRpeEstimate(p).value : null;
  const estimates = ratingHistory.map((h) => h.estimate).filter((v) => v !== null);
  const lines = [
    `# Fatigue session report: athlete ${p.subject_id}`,
    '',
    `Generated ${new Date().toLocaleString()}`,
    '',
    '## Snapshot',
    `- Exercise: ${exerciseName(p)} at ${p.intensity_band}% of maximum strength`,
    `- Time into exercise: ${Math.round(p.elapsed_seconds)} s`,
    `- Perceived effort now: ${est === null ? 'still calibrating' : `${est.toFixed(1)} of 20 (${effortWord(est).toLowerCase()}), ${effortStatus(est).word.toLowerCase()}`}`,
  ];
  if (estimates.length) {
    lines.push(`- Range this session: ${Math.min(...estimates).toFixed(1)} to ${Math.max(...estimates).toFixed(1)} of 20 over ${estimates.length} readings`);
  }
  lines.push(`- ${describeEstimateMethod(p)}`);
  if (lastAdvice) lines.push('', '## Latest safety check', `- ${lastAdvice.status} at ${lastAdvice.time}: ${lastAdvice.text}`);
  if (transcript.length) {
    lines.push('', '## Coach conversation');
    transcript.forEach((t) => lines.push(`- **Q (at ${t.reading} s):** ${t.question}`, `  **A${t.source === 'llm' ? ' (AI-written, figures verified)' : ' (rule-based)'}:** ${t.answer.replace(/\n/g, ' ')}`));
  }
  lines.push('', '---', 'Fatigue assessment for research and ergonomic use. Not medical advice.');
  return lines.join('\n');
}

function initExport() {
  $('exportReport').addEventListener('click', () => {
    if (!lastPayload) return;
    const blob = new Blob([buildReport()], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fatigue-report-${lastPayload.subject_id}.md`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
}

// ---------------------------------------------------------------------------
// Start-up
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  initThemeToggle();
  initCharts();
  initNavigation();
  initPlaybackToggle();
  initSafetyCheck();
  initCoachChat();
  initExport();
  initSubjectSelector();
  loadEngineStatus();
  loadSubjects();
  refreshIcons();
});
