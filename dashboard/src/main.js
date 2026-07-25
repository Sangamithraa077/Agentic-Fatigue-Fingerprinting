import './style.css';
import Chart from 'chart.js/auto';

// State Management
let isPlaying = true;
let activeTab = 'signals';
let streamInterval = null;
let sampleCounter = 0;

// Chart Instances
const charts = {};

// Synthetic Data Generators for Multimodal WSD4FEDSRM Stream
function generateSignalData(count, amplitude, freq, noise = 0.1) {
  const data = [];
  for (let i = 0; i < count; i++) {
    const t = i * 0.05;
    const val = amplitude * Math.sin(2 * Math.PI * freq * t) + (Math.random() - 0.5) * noise;
    data.push(val);
  }
  return data;
}

// Initialize Charts
function initCharts() {
  const commonOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } },
      tooltip: { backgroundColor: '#1e2638', titleColor: '#f0f4fc', bodyColor: '#94a3b8' }
    },
    scales: {
      x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } },
      y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#64748b' } }
    }
  };

  // 1. EMG Chart (6 channels)
  const labels50 = Array.from({ length: 50 }, (_, i) => `${(i * 0.01).toFixed(2)}s`);
  const emgCtx = document.getElementById('emgChart').getContext('2d');
  charts.emg = new Chart(emgCtx, {
    type: 'line',
    data: {
      labels: labels50,
      datasets: [
        { label: 'Ant. Deltoid', data: generateSignalData(50, 0.4, 8, 0.15), borderColor: '#00f0ff', borderWidth: 1.5, pointRadius: 0 },
        { label: 'Infraspinatus', data: generateSignalData(50, 0.3, 10, 0.12), borderColor: '#3b82f6', borderWidth: 1.5, pointRadius: 0 },
        { label: 'Lat. Dorsi', data: generateSignalData(50, 0.25, 6, 0.1), borderColor: '#8b5cf6', borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: {
      ...commonOptions,
      animation: false,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, min: -0.8, max: 0.8, title: { display: true, text: 'mV', color: '#64748b' } }
      }
    }
  });

  // 2. IMU Chart
  const imuCtx = document.getElementById('imuChart').getContext('2d');
  charts.imu = new Chart(imuCtx, {
    type: 'line',
    data: {
      labels: labels50,
      datasets: [
        { label: 'Acc X', data: generateSignalData(50, 2.5, 1.2, 0.2), borderColor: '#ef4444', borderWidth: 1.5, pointRadius: 0 },
        { label: 'Acc Y', data: generateSignalData(50, 1.8, 1.2, 0.2), borderColor: '#10b981', borderWidth: 1.5, pointRadius: 0 },
        { label: 'Acc Z', data: generateSignalData(50, 9.8, 0.5, 0.1), borderColor: '#f59e0b', borderWidth: 1.5, pointRadius: 0 }
      ]
    },
    options: {
      ...commonOptions,
      animation: false,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, title: { display: true, text: 'm/s²', color: '#64748b' } }
      }
    }
  });

  // 3. PPG Chart
  const ppgCtx = document.getElementById('ppgChart').getContext('2d');
  charts.ppg = new Chart(ppgCtx, {
    type: 'line',
    data: {
      labels: labels50,
      datasets: [
        { label: 'PPG Pulse Wave', data: generateSignalData(50, 1.2, 1.5, 0.05), borderColor: '#a855f7', borderWidth: 2, pointRadius: 0, fill: true, backgroundColor: 'rgba(168, 85, 247, 0.1)' }
      ]
    },
    options: {
      ...commonOptions,
      animation: false,
      scales: {
        ...commonOptions.scales,
        y: { ...commonOptions.scales.y, title: { display: true, text: 'V', color: '#64748b' } }
      }
    }
  });

  // 4. Fatigue Progression Curve (Borg RPE over time)
  const progressionCtx = document.getElementById('fatigueProgressionChart').getContext('2d');
  const timeLabels = ['0s', '30s', '60s', '90s', '120s', '150s', '180s', '210s'];
  charts.progression = new Chart(progressionCtx, {
    type: 'line',
    data: {
      labels: timeLabels,
      datasets: [
        { label: 'Ground Truth Borg RPE', data: [6, 8, 10, 12, 14, 16, 18, 20], borderColor: '#64748b', borderDash: [5, 5], borderWidth: 2, pointRadius: 4 },
        { label: 'Deep TCN Model Prediction', data: [6.1, 7.9, 10.2, 12.5, 14.2, 16.4, 17.8, 19.5], borderColor: '#ef4444', borderWidth: 3, pointRadius: 4, backgroundColor: 'rgba(239, 68, 68, 0.1)', fill: true },
        { label: 'Baseline Random Forest', data: [6.0, 7.2, 9.1, 11.0, 13.1, 14.8, 16.5, 18.1], borderColor: '#3b82f6', borderWidth: 2, pointRadius: 3 }
      ]
    },
    options: commonOptions
  });

  // 5. Muscle Spectral Frequency Breakdown
  const spectralCtx = document.getElementById('muscleSpectralChart').getContext('2d');
  charts.spectral = new Chart(spectralCtx, {
    type: 'bar',
    data: {
      labels: ['Ant. Deltoid', 'Infraspinatus', 'Lat. Dorsi', 'Pect. Major', 'Post. Deltoid', 'Upp. Trapezius'],
      datasets: [
        { label: 'Baseline MDF (Hz)', data: [110, 98, 105, 115, 102, 120], backgroundColor: 'rgba(59, 130, 246, 0.6)' },
        { label: 'Current Fatigue MDF (Hz)', data: [85.5, 76.2, 89.0, 102.1, 88.4, 98.7], backgroundColor: 'rgba(239, 68, 68, 0.8)' }
      ]
    },
    options: commonOptions
  });

  // 6. Global SHAP Bar Chart
  const shapBarCtx = document.getElementById('shapBarChart').getContext('2d');
  charts.shapBar = new Chart(shapBarCtx, {
    type: 'bar',
    indexAxis: 'y',
    data: {
      labels: [
        'EMG_MDF_AntDeltoid',
        'IMU_Jerk_UpperArm',
        'EMG_RMS_Infraspinatus',
        'PPG_HeartRate',
        'UpperArm_Length',
        'MVIC_Baseline_Force',
        'IMU_Acc_Std_Forearm'
      ],
      datasets: [
        { label: 'Mean |SHAP Value| (RPE Impact)', data: [2.84, 1.62, 1.45, 1.12, 0.95, 0.78, 0.54], backgroundColor: '#00f0ff' }
      ]
    },
    options: commonOptions
  });

  // 7. Local SHAP Waterfall
  const shapWaterfallCtx = document.getElementById('shapWaterfallChart').getContext('2d');
  charts.shapWaterfall = new Chart(shapWaterfallCtx, {
    type: 'bar',
    indexAxis: 'y',
    data: {
      labels: [
        'EMG_MDF_AntDeltoid (Shift -24.5Hz)',
        'IMU_Jerk_UpperArm (+4.2 m/s³)',
        'EMG_RMS_Infraspinatus (+0.18 mV)',
        'PPG_HeartRate (118 BPM)',
        'UpperArm_Length (34.2 cm)'
      ],
      datasets: [
        {
          label: 'SHAP Contribution to Borg RPE',
          data: [+3.2, +1.8, +1.1, -0.4, -0.3],
          backgroundColor: (ctx) => (ctx.raw > 0 ? '#ef4444' : '#10b981')
        }
      ]
    },
    options: commonOptions
  });

  // 8. LOSO Comparison Chart
  const losoCtx = document.getElementById('losoComparisonChart').getContext('2d');
  const subjects = Array.from({ length: 15 }, (_, i) => `P${(i + 1).toString().padStart(2, '0')}`);
  charts.loso = new Chart(losoCtx, {
    type: 'bar',
    data: {
      labels: subjects,
      datasets: [
        { label: 'Generic Population Model RMSE', data: [1.85, 2.10, 1.95, 2.40, 1.75, 2.05, 2.25, 1.90, 2.15, 1.80, 2.30, 1.98, 2.02, 1.88, 2.12], backgroundColor: 'rgba(239, 68, 68, 0.5)' },
        { label: 'Personalized LOSO Adaptor RMSE', data: [0.92, 1.05, 0.88, 1.15, 0.85, 0.98, 1.10, 0.90, 1.02, 0.82, 1.12, 0.94, 0.96, 0.89, 1.01], backgroundColor: 'rgba(16, 185, 129, 0.8)' }
      ]
    },
    options: commonOptions
  });
}

// Live Stream Loop
function startStream() {
  if (streamInterval) clearInterval(streamInterval);
  streamInterval = setInterval(() => {
    if (!isPlaying) return;
    sampleCounter++;

    // Update EMG data
    const emgChart = charts.emg;
    if (emgChart) {
      emgChart.data.datasets.forEach((ds, idx) => {
        ds.data.shift();
        const t = sampleCounter * 0.05;
        const amp = 0.3 + idx * 0.1 + (sampleCounter % 20 > 10 ? 0.2 : 0);
        ds.data.push(amp * Math.sin(2 * Math.PI * (8 + idx * 2) * t) + (Math.random() - 0.5) * 0.2);
      });
      emgChart.update('none');
    }

    // Update IMU data
    const imuChart = charts.imu;
    if (imuChart) {
      imuChart.data.datasets.forEach((ds, idx) => {
        ds.data.shift();
        const t = sampleCounter * 0.05;
        const offset = idx === 2 ? 9.8 : 0;
        ds.data.push(offset + Math.sin(2 * Math.PI * 1.2 * t + idx) * 2 + (Math.random() - 0.5) * 0.3);
      });
      imuChart.update('none');
    }

    // Update PPG data
    const ppgChart = charts.ppg;
    if (ppgChart) {
      const ppgDs = ppgChart.data.datasets[0];
      ppgDs.data.shift();
      const t = sampleCounter * 0.05;
      ppgDs.data.push(1.0 + Math.sin(2 * Math.PI * 1.8 * t) * 0.8 + (Math.random() - 0.5) * 0.05);
      ppgChart.update('none');
    }

    // Dynamic metrics perturbation
    const emgVal = (0.45 + Math.sin(sampleCounter * 0.1) * 0.08).toFixed(3);
    const jerkVal = (12.0 + Math.cos(sampleCounter * 0.1) * 1.5).toFixed(1);
    const hrVal = Math.floor(115 + Math.sin(sampleCounter * 0.05) * 6);

    const emgEl = document.getElementById('emgRmsVal');
    const imuEl = document.getElementById('imuJerkVal');
    const ppgEl = document.getElementById('ppgHrVal');

    if (emgEl) emgEl.innerHTML = `${emgVal} <span class="unit">mV</span>`;
    if (imuEl) imuEl.innerHTML = `${jerkVal} <span class="unit">m/s³</span>`;
    if (ppgEl) ppgEl.innerHTML = `${hrVal} <span class="unit">BPM</span>`;

  }, 100);
}

// Navigation Tab Handler
function initNavigation() {
  const tabs = document.querySelectorAll('.nav-item');
  const pageTitle = document.getElementById('pageTitle');
  const pageSubtitle = document.getElementById('pageSubtitle');

  const titles = {
    signals: { title: 'Multimodal Sensor Stream Analysis', sub: 'Real-time signal inspection, artifact filtering, and synchronized feature extraction' },
    predictions: { title: 'Personalized Fatigue Fingerprint', sub: 'Borg RPE / KSS score estimation via Deep Temporal Convolutional Networks (TCN)' },
    explainability: { title: 'SHAP Explainable AI Diagnostics', sub: 'Feature-level attribution vectors explaining muscle degradation drivers' },
    personalization: { title: 'Leave-One-Subject-Out (LOSO) Domain Adaptation', sub: 'Subject profile anthropometrics and personal baseline vs population baseline' },
    agents: { title: 'Autonomous Agentic Oversight System', sub: 'Multi-agent monitoring loop with threshold safety triggers and clinical intervention logs' }
  };

  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((t) => t.classList.remove('active'));
      tab.classList.add('active');

      const targetTab = tab.dataset.tab;
      activeTab = targetTab;

      document.querySelectorAll('.tab-content').forEach((content) => {
        content.classList.remove('active');
      });

      const activeEl = document.getElementById(`tab-${targetTab}`);
      if (activeEl) activeEl.classList.add('active');

      if (titles[targetTab]) {
        pageTitle.textContent = titles[targetTab].title;
        pageSubtitle.textContent = titles[targetTab].sub;
      }
    });
  });
}

// Playback Toggle
function initPlaybackToggle() {
  const btn = document.getElementById('togglePlayback');
  const text = document.getElementById('playbackBtnText');

  btn.addEventListener('click', () => {
    isPlaying = !isPlaying;
    if (isPlaying) {
      text.textContent = 'Pause Stream';
      btn.style.borderColor = 'transparent';
    } else {
      text.textContent = 'Resume Stream';
      btn.style.borderColor = 'var(--color-warning)';
    }
  });
}

// Agent Loop Simulation
function initAgentLoop() {
  const logBox = document.getElementById('agentLogs');
  const triggerBtn = document.getElementById('triggerAgentCheck');
  const slider = document.getElementById('alertThreshold');
  const threshVal = document.getElementById('thresholdVal');

  slider.addEventListener('input', (e) => {
    threshVal.textContent = `${e.target.value} (Hard Exertion)`;
  });

  function addLog(tag, tagClass, text) {
    const time = new Date().toISOString().split('T')[1].slice(0, 8);
    const entry = document.createElement('div');
    entry.className = 'log-entry';
    entry.innerHTML = `<span class="log-timestamp">[${time}]</span> <span class="log-tag ${tagClass}">${tag}</span> ${text}`;
    logBox.prepend(entry);
  }

  // Initial Logs
  addLog('SENTINEL', 'tag-sentinel', 'System initialized. Baseline Borg RPE: 6.0. Monitoring 6 EMG & 6 IMU streams.');
  addLog('PERSONAL', 'tag-personal', 'Loaded Subject P03 profile (Male, 26y, 183cm, UpperArm: 34.2cm). LOSO adaptor active.');

  triggerBtn.addEventListener('click', () => {
    const currentRpe = parseFloat(document.getElementById('currentRpe').textContent);
    const threshold = parseFloat(slider.value);

    addLog('SENTINEL', 'tag-sentinel', `Agent cycle triggered. Evaluated RPE: ${currentRpe} against threshold: ${threshold}`);

    if (currentRpe >= threshold) {
      addLog('SENTINEL', 'tag-sentinel', `⚠️ ALERT: Borg RPE (${currentRpe}) exceeded threshold (${threshold})! Muscle fatigue shift detected.`);
      addLog('EXPLAIN', 'tag-explain', 'SHAP Analysis: Ant. Deltoid MDF dropped -24.5Hz (Impact: +3.2 RPE). Recommending 60s rest interval.');
    } else {
      addLog('SENTINEL', 'tag-sentinel', '✅ Subject operating within safe muscle exertion limits.');
    }
  });
}

// Subject Selector Handler
function initSubjectSelector() {
  const select = document.getElementById('subjectSelect');
  select.addEventListener('change', (e) => {
    const val = e.target.value;
    const logBox = document.getElementById('agentLogs');
    if (logBox) {
      const time = new Date().toISOString().split('T')[1].slice(0, 8);
      const entry = document.createElement('div');
      entry.className = 'log-entry';
      entry.innerHTML = `<span class="log-timestamp">[${time}]</span> <span class="log-tag tag-personal">PERSONAL</span> Switched active context to ${val}. Calibrating LOSO feature weights...`;
      logBox.prepend(entry);
    }
  });
}

// Document Ready Initialization
document.addEventListener('DOMContentLoaded', () => {
  initCharts();
  initNavigation();
  initPlaybackToggle();
  initAgentLoop();
  initSubjectSelector();
  startStream();
});
