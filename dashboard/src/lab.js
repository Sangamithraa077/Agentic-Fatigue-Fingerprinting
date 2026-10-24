// Session lab -- replay of a recorded exercise with the dataset pipeline's estimates.
//
// Everything shown comes from /api/pipeline/*: stored pipeline outputs (estimates out-of-sample by person, the
// athlete's own rating, feature trends, drivers, quality flags). The explanation is written by the backend's
// explanation layer in a background job, labelled as generated text, and every number in it is checked against the
// figures on this page. Nothing here is computed or invented in the browser beyond smoothing-free display maths.
//
// Shares helpers with main.js (esc, refreshIcons, getChartPalette, borgZonesPlugin, API_BASE, WARN_LEVEL, STOP_LEVEL).

(() => {
  const $ = (id) => document.getElementById(id);
  const S = {
    ready: false, people: [], sid: null, task: null, data: null, step: -1, model: 'generalised', playing: false,
    speed: 10, timer: null, charts: {}, trends: new Set(), job: null, lastExplainStep: -99, firedWarn: false,
    firedStop: false, explainBusy: false,
  };
  const SERIES_COLORS = ['#4cc9f0', '#a78bfa', '#fbbf24', '#34d399', '#fb7185', '#6d8cff', '#f97316', '#94a3b8'];
  const ASK_EXAMPLES = [
    'What was the highest effort?', 'When did effort reach the warning level?', 'How close was the estimate to my own rating?',
    'Is the personalised model better than the generalised one?', 'What changed the most from my start?',
    'How good was the data quality?', 'Which exercise lasted longest?',
  ];

  const api = async (path, opts) => {
    const res = await fetch(`${API_BASE}${path}`, opts);
    if (!res.ok) {
      let detail = res.statusText;
      try { detail = (await res.json()).detail || detail; } catch (_) { /* keep status text */ }
      throw new Error(detail);
    }
    return res.json();
  };
  const post = (path, body) => api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  const showError = (msg) => { const b = $('labError'); b.textContent = msg; b.hidden = !msg; };
  const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '--' : Number(v).toFixed(d));
  const word = (v) => (typeof effortWord === 'function' ? effortWord(v) : '');
  const statusFor = (v) => (v >= STOP_LEVEL ? ['Critical', 'tone-alert'] : v >= WARN_LEVEL ? ['Fatigued', 'tone-moderate'] : v >= 11 ? ['Working', 'tone-good'] : ['Fresh', 'tone-good']);
  const cur = () => (S.data && S.step >= 0 ? S.data.windows_data[S.step] : null);
  const estimateOf = (w) => (S.model === 'personalised' && !w.calibrating ? w.smoothed_personalised : w.smoothed_generalised);

  // ---------------------------------------------------------------- setup
  async function init() {
    if (S.ready) return;
    S.ready = true;
    try {
      const st = await api('/api/pipeline/status');
      if (!st.pipeline_tables) throw new Error('The pipeline tables are not available. Run the dataset pipeline (see docs/dataset_pipeline_modules_1_7.md).');
      $('labSourceNote').textContent += st.llm_enabled
        ? ` Explanations are written by ${st.llm_backend} and checked before they are shown.`
        : ' No language model is configured, so explanations are rule-based text built only from the numbers (the page works the same).';
      S.people = (await api('/api/pipeline/participants')).participants;
    } catch (e) { showError(e.message); S.ready = false; return; }
    $('labPerson').innerHTML = S.people.map((p) => `<option value="${p.sid}">${esc(p.id)} · ${esc(p.sex)}, ${esc(p.age)} · quality ${esc(fmt(p.quality_score, 0))}</option>`).join('');
    $('labAskChips').innerHTML = ASK_EXAMPLES.map((q) => `<button type="button" class="chip chip-btn" data-q="${esc(q)}">${esc(q)}</button>`).join('');
    bind();
    await loadSessions(Number($('labPerson').value));
    loadResults();
  }

  function bind() {
    $('labPerson').addEventListener('change', (e) => loadSessions(Number(e.target.value)));
    $('labTask').addEventListener('change', (e) => loadSession(S.sid, e.target.value));
    $('labPlay').addEventListener('click', () => (S.playing ? pause() : play()));
    $('labRestart').addEventListener('click', () => { pause(); resetRun(); render(-1); });
    $('labScrub').addEventListener('input', (e) => { pause(); render(Number(e.target.value)); });
    $('labModel').addEventListener('click', (e) => {
      const b = e.target.closest('[data-model]'); if (!b) return;
      S.model = b.dataset.model;
      $('labModel').querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('active', x === b));
      render(S.step);
    });
    $('labSpeed').addEventListener('click', (e) => {
      const b = e.target.closest('[data-speed]'); if (!b) return;
      S.speed = Number(b.dataset.speed);
      $('labSpeed').querySelectorAll('.seg-btn').forEach((x) => x.classList.toggle('active', x === b));
      if (S.playing) { clearInterval(S.timer); S.timer = setInterval(tick, 10000 / S.speed); }
    });
    $('labExplainNow').addEventListener('click', () => requestExplanation({ force: true }));
    $('labAskForm').addEventListener('submit', (e) => { e.preventDefault(); ask($('labAskInput').value.trim()); });
    $('labAskChips').addEventListener('click', (e) => { const b = e.target.closest('[data-q]'); if (b) { $('labAskInput').value = b.dataset.q; ask(b.dataset.q); } });
    $('labTrendChips').addEventListener('click', (e) => {
      const b = e.target.closest('[data-trend]'); if (!b) return;
      const k = b.dataset.trend;
      S.trends.has(k) ? S.trends.delete(k) : S.trends.add(k);
      b.classList.toggle('on', S.trends.has(k));
      drawTrends();
    });
    document.addEventListener('keydown', (e) => {
      if (document.body.dataset.tab !== 'lab' || e.key !== ' ' || /^(INPUT|TEXTAREA|SELECT|BUTTON|A)$/.test(document.activeElement?.tagName || '')) return;
      e.preventDefault(); S.playing ? pause() : play();
    });
  }

  async function loadSessions(sid) {
    S.sid = sid;
    try {
      const { sessions } = await api(`/api/pipeline/participants/${sid}/sessions`);
      $('labTask').innerHTML = sessions.map((s) => `<option value="${esc(s.task)}">${esc(s.label)} · ${esc(s.duration_s.toFixed(0))} s</option>`).join('');
      $('labTask').disabled = false;
      await loadSession(sid, sessions[0].task);
    } catch (e) { showError(e.message); }
  }

  async function loadSession(sid, task) {
    pause(); showError('');
    try { S.data = await api(`/api/pipeline/session/${sid}/${task}`); } catch (e) { showError(e.message); return; }
    S.task = task;
    const n = S.data.windows_data.length;
    $('labScrub').max = n - 1; $('labScrub').disabled = false;
    $('labExplainNow').disabled = false;
    const base = `${API_BASE}/api/pipeline/report/${sid}/${task}`;
    $('labReportHtml').href = `${base}?format=html`; $('labReportMd').href = `${base}?format=md`;
    $('labReportMd').setAttribute('download', `session_report_P${String(sid).padStart(2, '0')}_${task}.md`);
    ['labReportHtml', 'labReportMd'].forEach((id) => $(id).removeAttribute('aria-disabled'));
    $('labTrendChips').innerHTML = Object.entries(S.data.trend_labels).map(([k, label], i) =>
      `<button type="button" class="chip chip-btn ${i < 4 ? 'on' : ''}" data-trend="${esc(k)}"><span class="sw" style="background:${SERIES_COLORS[i]}"></span>${esc(label)}</button>`).join('');
    S.trends = new Set(Object.keys(S.data.trend_labels).slice(0, 4));
    renderProfile();
    resetRun();
    buildCharts();
    render(-1);
  }

  function resetRun() {
    S.step = -1; S.lastExplainStep = -99; S.firedWarn = false; S.firedStop = false; S.job = null; S.explainBusy = false;
    $('labExplainBody').innerHTML = '<p class="muted">No explanation yet. The estimate keeps running without one.</p>';
    $('labExplainMeta').hidden = true; $('labAlerts').innerHTML = '';
    $('labAskAnswer').innerHTML = '';
  }

  // ---------------------------------------------------------------- playback
  function play() {
    if (!S.data) return;
    if (S.step >= S.data.windows_data.length - 1) { resetRun(); render(-1); }
    S.playing = true;
    $('labPlay').innerHTML = '<i data-lucide="pause"></i><span>Pause</span>'; refreshIcons();
    S.timer = setInterval(tick, 10000 / S.speed);
    tick();
  }
  function pause() {
    S.playing = false; clearInterval(S.timer);
    const b = $('labPlay'); if (b) { b.innerHTML = '<i data-lucide="play"></i><span>Play</span>'; refreshIcons(); }
  }
  function tick() {
    const next = S.step + 1;
    if (next >= S.data.windows_data.length) { pause(); return; }
    render(next, { auto: true });
    if (next === S.data.windows_data.length - 1) { pause(); requestExplanation({ concluded: true }); }
  }

  // ---------------------------------------------------------------- rendering
  function render(step, { auto = false } = {}) {
    S.step = step;
    $('labScrub').value = Math.max(step, 0);
    const w = cur();
    $('labClock').textContent = w ? `${w.t.toFixed(0)} s` : '0 s';
    drawGauge(w);
    drawRpeChart();
    drawTrends();
    renderDrivers(w);
    renderQuality(w);
    if (!w) return;
    const est = estimateOf(w);
    $('labSelf').textContent = `${fmt(w.self_report, 0)} · ${word(w.self_report)}`;
    const d = est - w.self_report;
    $('labDiff').textContent = `${d >= 0 ? '+' : ''}${fmt(d)} points`;
    if (auto) maybeAutoExplain(w, step);
  }

  function drawGauge(w) {
    const fill = $('labGaugeFill'), needle = $('labNeedle');
    if (!w) {
      $('labGaugeValue').textContent = '--'; fill.setAttribute('d', 'M 20 110 A 90 90 0 0 1 20 110');
      needle.setAttribute('x2', 30); needle.setAttribute('y2', 110);
      $('labStatusText').textContent = 'Ready'; $('labStatus').className = 'status-pill tone-neutral';
      $('labCalChip').textContent = 'Baseline: waiting'; $('labGaugeSub').textContent = 'Pick a participant and press play.';
      $('labSelf').textContent = '--'; $('labDiff').textContent = '--';
      $('labQuality').textContent = S.data && S.data.quality_score != null ? `${fmt(S.data.quality_score, 0)} / 100` : '--';
      return;
    }
    const v = Math.min(20, Math.max(6, estimateOf(w)));
    const th = Math.PI * (1 - (v - 6) / 14);
    const R = 90, cx = 110, cy = 110;
    fill.setAttribute('d', `M ${cx - R} ${cy} A ${R} ${R} 0 0 1 ${(cx + R * Math.cos(th)).toFixed(1)} ${(cy - R * Math.sin(th)).toFixed(1)}`);
    needle.setAttribute('x2', (cx + 78 * Math.cos(th)).toFixed(1)); needle.setAttribute('y2', (cy - 78 * Math.sin(th)).toFixed(1));
    const pt = S.data.personalisation_test;
    $('labPersonalNote').hidden = S.model !== 'personalised' || !pt;
    if (pt) {
      $('labPersonalNote').textContent = pt.significant_gain
        ? `Tested on ${pt.n_people} people: scaling to the athlete's own first ${pt.calibration_seconds} s lowered the error from ${pt.mae_generalised.toFixed(2)} to ${pt.mae_personalised.toFixed(2)} points.`
        : `Tested on ${pt.n_people} people: scaling to the athlete's own first ${pt.calibration_seconds} s was no more accurate than the generalised estimate on average (typical error ${pt.mae_personalised.toFixed(2)} vs ${pt.mae_generalised.toFixed(2)} Borg points, a difference that could be chance; closer for ${pt.people_improved} of ${pt.n_people} people). Treat it as an alternative view, not a better one.`;
    }
    const [wordTxt, tone] = statusFor(v);
    fill.setAttribute('class', `g-fill ${tone}`);
    $('labGaugeValue').textContent = v.toFixed(1);
    $('labStatusText').textContent = wordTxt; $('labStatus').className = `status-pill ${tone}`;
    const k = S.data.calibration_windows;
    $('labCalChip').textContent = w.baseline_forming ? 'Forming baseline (first 20 s)' : w.calibrating ? `Calibrating ${w.win + 1} of ${k} windows` : 'Calibrated';
    $('labGaugeSub').textContent = `${word(v)} at ${w.t.toFixed(0)} s into ${S.data.label.toLowerCase()}`;
    $('labQuality').textContent = S.data.quality_score != null ? `${fmt(S.data.quality_score, 0)} / 100` : '--';
  }

  const seriesFor = (key, upto) => S.data.windows_data.slice(0, upto + 1).map((w) => ({ x: w.t, y: w[key] }));

  function buildCharts() {
    Object.values(S.charts).forEach((c) => c && c.destroy());
    const p = getChartPalette();
    const font = { family: 'Inter', size: 11 };
    const base = (yTitle) => ({
      responsive: true, maintainAspectRatio: false, animation: false, parsing: false,
      interaction: { mode: 'index', intersect: false },
      scales: {
        x: { type: 'linear', min: 0, max: Math.ceil(S.data.summary.duration_s / 10) * 10, title: { display: true, text: 'seconds', color: p.muted, font }, ticks: { color: p.muted, font }, grid: { color: p.grid } },
        y: { ...yTitle, ticks: { color: p.muted, font }, grid: { color: p.grid } },
      },
      plugins: { legend: { labels: { color: p.text, font, boxWidth: 12 } },
        tooltip: { backgroundColor: p.tooltipBg, borderColor: p.tooltipBorder, borderWidth: 1, titleColor: p.tooltipTitle, bodyColor: p.tooltipBody } },
    });
    const o1 = base({ min: 5, max: 21, title: { display: true, text: 'Borg effort (6 to 20)', color: p.muted, font } });
    S.charts.rpe = new Chart($('labRpeChart'), {
      type: 'line', plugins: typeof borgZonesPlugin !== 'undefined' ? [borgZonesPlugin] : [], options: o1,
      data: { datasets: [
        { label: 'Athlete\'s own rating', data: [], borderColor: p.text, backgroundColor: p.text, stepped: 'before', borderWidth: 2, pointRadius: 3 },
        { label: 'Generalised estimate', data: [], borderColor: p.cyan, backgroundColor: p.cyan, borderWidth: 2, pointRadius: 0, tension: 0.25 },
        { label: 'Personalised estimate', data: [], borderColor: p.purple, backgroundColor: p.purple, borderWidth: 2, borderDash: [6, 3], pointRadius: 0, tension: 0.25 },
      ] },
    });
    const o2 = base({ title: { display: true, text: '% change from own start', color: p.muted, font } });
    S.charts.trend = new Chart($('labTrendChart'), { type: 'line', options: o2, data: { datasets: [] } });
  }

  function drawRpeChart() {
    const c = S.charts.rpe; if (!c) return;
    const n = S.step;
    const personalised = S.data.windows_data.slice(0, n + 1).map((w) => ({ x: w.t, y: w.calibrating ? null : w.smoothed_personalised }));
    c.data.datasets[0].data = seriesFor('self_report', n);
    c.data.datasets[1].data = seriesFor('smoothed_generalised', n);
    c.data.datasets[2].data = personalised;
    c.data.datasets[1].borderWidth = S.model === 'generalised' ? 3 : 1.5;
    c.data.datasets[2].borderWidth = S.model === 'personalised' ? 3 : 1.5;
    c.update('none');
  }

  function drawTrends() {
    const c = S.charts.trend; if (!c || !S.data) return;
    const keys = Object.keys(S.data.trend_labels);
    c.data.datasets = keys.filter((k) => S.trends.has(k)).map((k) => ({
      label: S.data.trend_labels[k], borderColor: SERIES_COLORS[keys.indexOf(k)], backgroundColor: SERIES_COLORS[keys.indexOf(k)],
      borderWidth: 2, pointRadius: 0, tension: 0.25,
      data: S.data.windows_data.slice(0, S.step + 1).filter((w) => w.trends[k] != null).map((w) => ({ x: w.t, y: w.trends[k] * 100 })),
    }));
    c.update('none');
  }

  function renderDrivers(w) {
    const ul = $('labDrivers');
    if (!w || !w.drivers.length) { ul.innerHTML = `<li class="muted">${w ? 'The first 20 seconds form this athlete\'s own baseline, so there is nothing to compare yet.' : 'Press play.'}</li>`; return; }
    const max = Math.max(...w.drivers.map((d) => Math.abs(d.impact)), 0.01);
    ul.innerHTML = w.drivers.map((d) => `<li><span class="d-name">${esc(d.label)}</span>
      <span class="d-bar ${d.impact >= 0 ? 'up' : 'down'}"><i style="width:${(Math.abs(d.impact) / max * 100).toFixed(0)}%"></i></span>
      <span class="d-val">${d.impact >= 0 ? 'raises' : 'lowers'} ${Math.abs(d.impact).toFixed(2)}</span></li>`).join('');
  }

  function renderQuality(w) {
    const el = $('labQualityList');
    if (!w || !w.quality_flags.length) { el.innerHTML = '<span class="chip chip-quiet">All channels clean in this window</span>'; return; }
    el.innerHTML = w.quality_flags.map((f) => `<span class="chip chip-up">${esc(f)}</span>`).join('');
  }

  function renderProfile() {
    const p = S.people.find((x) => x.sid === S.sid);
    const b = S.data.baseline || {};
    const rows = [['Age / sex', p ? `${p.age} · ${p.sex}` : '--'], ['Data-quality score', S.data.quality_score != null ? `${fmt(S.data.quality_score, 0)} / 100` : '--'],
      ['Session length', `${S.data.summary.duration_s.toFixed(0)} s, ${S.data.summary.windows} windows`],
      ['Starting pulse', b.hr_bpm != null ? `${b.hr_bpm.toFixed(0)} beats/min` : '--'],
      ['Starting rotator-cuff signal speed', b.emg_infraspinatus_mdf != null ? `${b.emg_infraspinatus_mdf.toFixed(0)} Hz` : '--'],
      ['Estimate error, this session', `${fmt(S.data.summary.mae_generalised, 2)} points`]];
    $('labProfile').innerHTML = rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${esc(v)}</dd></div>`).join('');
  }

  // ---------------------------------------------------------------- explanation (background job, never blocks playback)
  function maybeAutoExplain(w, step) {
    const est = estimateOf(w);
    const prev = S.data.windows_data[Math.max(0, step - 3)];
    const rise = est - estimateOf(prev);
    let reason = null;
    if (est >= STOP_LEVEL && !S.firedStop) { S.firedStop = true; S.firedWarn = true; reason = 'stop'; }
    else if (est >= WARN_LEVEL && !S.firedWarn) { S.firedWarn = true; reason = 'warning'; }
    else if (rise >= 2.5 && step - S.lastExplainStep >= 6) reason = 'rise';
    if (est >= WARN_LEVEL) alertChip(est >= STOP_LEVEL ? 'stop' : 'warning', w.t);
    if (reason) requestExplanation({});
  }

  function alertChip(level, t) {
    const el = $('labAlerts');
    if (el.querySelector(`[data-level="${level}"]`)) return;
    el.insertAdjacentHTML('beforeend', `<span class="chip ${level === 'stop' ? 'chip-bad' : 'chip-up'}" data-level="${level}">${level === 'stop' ? 'Stop level (18)' : 'Warning level (15)'} reached at ${t.toFixed(0)} s</span>`);
  }

  async function requestExplanation({ force = false, concluded = false, query = null }) {
    if (S.explainBusy && !force) return;
    const step = Math.max(S.step, 0);
    S.explainBusy = true; S.lastExplainStep = step;
    const body = $('labExplainBody');
    try {
      const res = await post('/api/pipeline/explain', { sid: S.sid, task: S.task, step, model: S.model, user_query: query, session_concluded: concluded, force });
      if (!res.triggered) { S.explainBusy = false; if (force) body.innerHTML = '<p class="muted">Nothing unusual to explain at this moment (estimate stays machine-only).</p>'; return; }
      body.innerHTML = '<p class="muted"><span class="spin"></span> Writing an explanation… the replay keeps running.</p>';
      const job = res.job_id;
      for (let i = 0; i < 60; i += 1) {
        await new Promise((r) => setTimeout(r, 350));
        const j = await api(`/api/pipeline/explain/${job}`);
        if (j.status === 'done') { showExplanation(j.result, res); break; }
        if (j.status === 'error') throw new Error(j.error);
      }
    } catch (e) { body.innerHTML = `<p class="muted">Could not write an explanation: ${esc(e.message)}. The estimate itself is unaffected.</p>`; }
    S.explainBusy = false;
  }

  function showExplanation(r, ctx) {
    const sums = ctx.summary || {};
    $('labExplainBody').innerHTML = `<p class="gen-label"><i data-lucide="bot"></i> Generated text · ${esc(r.generated_by)}</p><p class="explain-text">${esc(r.answer)}</p>`;
    const src = (r.sources || []).map((s) => `<li><span class="tag">${esc(s.chunk_id)}</span> ${esc(s.title)}<br><small>${esc(s.snippet)}</small></li>`).join('');
    const m = $('labExplainMeta'); m.hidden = false;
    m.innerHTML = `<div class="meta-row"><span class="chip chip-quiet">Triggered by: ${esc((ctx.trigger || '').toLowerCase().replace(/_/g, ' '))}</span>
      <span class="chip chip-quiet">${esc(r.delay_ms)} ms</span>
      <span class="chip chip-good">Numbers checked against this session</span>
      ${r.fallback_reason ? `<span class="chip chip-quiet">Fallback: ${esc(String(r.fallback_reason).replace(/_/g, ' '))}</span>` : ''}</div>
      <details><summary>Numbers it was allowed to use (${(r.numbers_used || []).length})</summary><p class="nums">${(r.numbers_used || []).map((n) => `<span class="chip chip-quiet">${esc(n)}</span>`).join(' ')}</p></details>
      <details><summary>Sources (${(r.sources || []).length})</summary><ul class="sources">${src || '<li class="muted">No notes retrieved.</li>'}</ul></details>
      <p class="fine">${esc((r.disclaimer || '').replace(/\*/g, '').trim())}</p>`;
    refreshIcons();
  }

  // ---------------------------------------------------------------- ask your data
  async function ask(q) {
    if (!q || !S.data) return;
    const out = $('labAskAnswer');
    out.innerHTML = '<p class="muted"><span class="spin"></span> Looking up…</p>';
    try {
      const r = await post('/api/pipeline/ask', { sid: S.sid, task: S.task, question: q });
      if (!r.matched) {
        out.innerHTML = `<p>${esc(r.answer)}</p><p class="muted">Try one of: ${(r.supported || []).map((s) => `<em>${esc(s.example)}</em>`).join(' · ')}</p>`;
      } else {
        out.innerHTML = `<p class="gen-label"><i data-lucide="database"></i> Read-only lookup · ${esc(r.matched.replace(/_/g, ' '))}</p><p class="explain-text">${esc(r.answer)}</p><p class="fine">${esc(r.source)}</p>`;
      }
      refreshIcons();
    } catch (e) { out.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }

  // ---------------------------------------------------------------- published results
  async function loadResults() {
    const el = $('labResultsBody');
    try {
      const r = await api('/api/pipeline/results');
      const table = (rows, cols, title, note) => !rows ? '' : `<h4 class="sub-h">${esc(title)}</h4>${note ? `<p class="fine">${esc(note)}</p>` : ''}
        <div class="tbl-wrap"><table class="tbl"><thead><tr>${cols.map(([, l]) => `<th>${esc(l)}</th>`).join('')}</tr></thead><tbody>
        ${rows.map((row) => `<tr>${cols.map(([k, , f]) => `<td>${esc(row[k] == null ? '--' : f ? f(row[k]) : row[k])}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
      const n2 = (v) => Number(v).toFixed(2), pc = (v) => `${(Number(v) * 100).toFixed(0)}%`;
      el.innerHTML =
        table(r.generalised_models, [['model', 'Model (leave-one-person-out, 34 people)'], ['MAE', 'Mean error (Borg points)', n2], ['within1', 'Within 1 point', pc], ['people_better_than_time_only', 'People better than elapsed-time-only']], 'Generalised models',
          'Elapsed time alone is a strong baseline because effort rises with time; the best sensor model beats it for 24 of 34 people.') +
        table(r.personalisation, [['model', 'Model (second half of each exercise)'], ['MAE', 'Mean error', n2], ['people_improved', 'People better than generalised'], ['wilcoxon_p', 'p-value', (v) => Number(v).toPrecision(2)]], 'Personalisation',
          'Calibrating on the first half of each exercise; only per-person normalisation helped.') +
        table(r.live_calibration, [['calibration_seconds', 'Calibration (s)'], ['MAE_generalised', 'Generalised error', n2], ['MAE_personalised', 'Personalised error', n2], ['people_improved', 'People improved'], ['wilcoxon_p', 'p-value', (v) => Number(v).toPrecision(2)]], 'Deployable personalisation',
          'Calibrating on the first 30 to 50 seconds, as a live system must.') +
        table(r.research_comparison, [['model', 'Model'], ['n_windows', 'Windows'], ['MAE', 'Mean error', n2]], 'Against the earlier research study', 'Scored on exactly the same windows.');
    } catch (e) { el.innerHTML = `<p class="muted">${esc(e.message)}</p>`; }
  }

  // Registered at load, not inside init(), so the first visit to the tab starts it.
  window.addEventListener('tabchange', (e) => {
    if (e.detail === 'lab') { init(); Object.values(S.charts).forEach((c) => c && c.resize()); } else pause();
  });
  if (document.body.dataset.tab === 'lab') init();
})();
