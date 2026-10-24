"""Modules 11 and 14: the live software core.

    DataSource (11.1)     FileReplaySource (cleaned recording, now) | SyntheticSource (hardware simulator, 14.4)
                          | SerialSource (real hardware, later - implement read_chunk against the contract)
    LivePipeline (11.2-11.5)   10 s windows -> the same features as the offline pipeline -> generalised estimate,
                          calibration on the first K windows -> personalised estimate, EMA-smoothed index,
                          alerts at 15 / 18, JSONL session log
    ExplanationService (11.6)  triggers the explanation layer in a background thread; the loop never waits for it
    validate_chunk (14.3)      the hardware contract, as code

The streaming path calls the same ``window_feats`` / baseline / model code as the offline tables, so replaying a
recording gives the offline predictions (checked in tests and by ``verify_replay``).
"""

import json
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m4_features import MIN_VALID_FRAC, N_BASE, window_feats
from ml.dataset_pipeline.personalise import COND, ZScoreModel

WINDOW_S = 10
CHUNK_S = 1
N_EMG, N_IMU = len(c.MUSCLES), 36
IMU_NAMES = [f"{k}_{tag}_{ax}" for tag in c.IMU_SITES.values() for k in ("acc", "gyr") for ax in "xyz"]
WARN_LEVEL, STOP_LEVEL = 15.0, 18.0

# ----------------------------------------------------------------------------- hardware contract (14.3)
CONTRACT = {
    "emg": {"channels": c.MUSCLES, "rate_hz": 1000, "unit": "mV", "abs_max": 10.0, "typical_max": 4.0,
            "csv_column": "EMG(mV)", "files": "EMG data/<muscle>.csv"},
    "imu": {"channels": IMU_NAMES, "rate_hz": 100, "units": {"acc": "m/s^2", "gyr": "rad/s"},
            "abs_max": {"acc": 160.0, "gyr": 35.0}, "csv_columns": "acc_x(m/s^2),acc_y,acc_z / gyr_x(rad/s),gyr_y,gyr_z",
            "files": "IMU data/<Site>/<acc|gyr>_<site>.csv"},
    "ppg": {"channels": ["ppg"], "rate_hz": 200, "unit": "V", "range": [0.0, 3.3], "csv_column": "PPG(Volt)",
            "files": "PPG data/ppg.csv"},
    "chunk_seconds": CHUNK_S,
    "window_seconds": WINDOW_S,
}


@dataclass
class Chunk:
    """One second of aligned samples from all three streams, in dataset units."""
    emg: np.ndarray            # (6, 1000) mV
    imu: np.ndarray            # (36, 100)  acc m/s^2 / gyr rad/s, order = IMU_NAMES
    ppg: np.ndarray            # (200,) V
    t0: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


def validate_chunk(ch: Chunk) -> List[str]:
    """Contract violations of one chunk (empty list = conforming)."""
    issues = []
    if ch.emg.shape != (N_EMG, CONTRACT["emg"]["rate_hz"] * CHUNK_S):
        issues.append(f"emg shape {ch.emg.shape}")
    if ch.imu.shape != (N_IMU, CONTRACT["imu"]["rate_hz"] * CHUNK_S):
        issues.append(f"imu shape {ch.imu.shape}")
    if ch.ppg.shape != (CONTRACT["ppg"]["rate_hz"] * CHUNK_S,):
        issues.append(f"ppg shape {ch.ppg.shape}")
    if issues:
        return issues
    if np.nanmax(np.abs(ch.emg)) > CONTRACT["emg"]["abs_max"]:
        issues.append("emg outside +-10 mV")
    acc = [i for i, n in enumerate(IMU_NAMES) if n.startswith("acc")]
    gyr = [i for i, n in enumerate(IMU_NAMES) if n.startswith("gyr")]
    if np.nanmax(np.abs(ch.imu[acc])) > CONTRACT["imu"]["abs_max"]["acc"]:
        issues.append("acc outside sensor range")
    if np.nanmax(np.abs(ch.imu[gyr])) > CONTRACT["imu"]["abs_max"]["gyr"]:
        issues.append("gyro outside sensor range")
    lo, hi = CONTRACT["ppg"]["range"]
    if np.nanmin(ch.ppg) < lo or np.nanmax(ch.ppg) > hi:
        issues.append("ppg outside 0-3.3 V")
    for name, arr in (("emg", ch.emg), ("imu", ch.imu), ("ppg", ch.ppg)):
        if np.isnan(arr).mean() > 0.2:
            issues.append(f"{name} more than 20% missing")
    return issues


# ----------------------------------------------------------------------------- data sources (11.1)
class DataSource(ABC):
    """Anything that yields one-second chunks in dataset units. Replay now, serial/Bluetooth later."""

    @abstractmethod
    def chunks(self) -> Iterator[Chunk]:
        ...


class FileReplaySource(DataSource):
    """Replays a cleaned recording (datasets/processed/clean_signals) as if it were arriving live."""

    def __init__(self, sid: int, task: str, speed: float = 0.0, clean_dir: Optional[Path] = None):
        self.sid, self.task, self.speed = sid, task, speed          # speed 0 = as fast as possible, 1 = real time
        z = np.load((clean_dir or c.CLEAN) / task / f"s{sid}.npz")
        assert list(z["imu_names"]) == IMU_NAMES, "cleaned recording does not follow the contract channel order"
        self.emg, self.imu, self.ppg = z["emg"].astype(float), z["imu"].astype(float), z["ppg"].astype(float)
        self.seconds = int(float(z["t_end"]))

    def chunks(self) -> Iterator[Chunk]:
        for s in range(self.seconds):
            t0 = time.perf_counter()
            yield Chunk(self.emg[:, s * 1000:(s + 1) * 1000], self.imu[:, s * 100:(s + 1) * 100],
                        self.ppg[s * 200:(s + 1) * 200], t0=float(s), meta={"source": "file_replay"})
            if self.speed > 0:
                time.sleep(max(0.0, CHUNK_S / self.speed - (time.perf_counter() - t0)))


class SerialSource(DataSource):
    """Placeholder for the wearable (ESP32 over USB-serial or BLE). Implement ``read_chunk`` so that it returns
    a Chunk conforming to CONTRACT; everything downstream is already hardware-agnostic."""

    def __init__(self, port: str = ""):
        self.port = port

    def read_chunk(self) -> Chunk:
        raise NotImplementedError("hardware reader not implemented yet; see docs/hardware_contract.md")

    def chunks(self) -> Iterator[Chunk]:
        while True:
            yield self.read_chunk()


class SyntheticSource(DataSource):
    """Hardware simulator (14.4): contract-conforming signals whose spectrum, amplitude, tempo and pulse drift
    with a known effort curve. It exercises the plumbing (rates, units, quality faults, latency); it is NOT recorded
    data and says nothing about accuracy on people."""

    def __init__(self, seconds: int = 160, seed: int = 0, fatigue_midpoint_s: Optional[float] = None, faults: Optional[List[dict]] = None):
        self.seconds, self.rng = seconds, np.random.default_rng(seed)
        self.mid = fatigue_midpoint_s or seconds * 0.55
        self.faults = faults or []
        self.phase = self.rng.uniform(0, 2 * np.pi, N_IMU)

    def effort(self, t: float) -> float:
        """Ground-truth effort on the Borg scale (6-20) at time t seconds."""
        return 6 + 14 / (1 + np.exp(-(t - self.mid) / (self.seconds * 0.16)))

    def _emg(self, t: float) -> np.ndarray:
        p = (self.effort(t) - 6) / 14
        fc, gain = 95 - 32 * p, 0.05 * (1 + 1.6 * p)            # median frequency falls, amplitude rises
        f = np.fft.rfftfreq(1000, 1 / 1000)
        shape = np.where((f >= 20) & (f <= 450), (f / fc) ** 2 * np.exp(-(f / fc) * 1.4), 0)
        out = np.empty((N_EMG, 1000))
        for j in range(N_EMG):
            spec = shape * (self.rng.normal(size=f.size) + 1j * self.rng.normal(size=f.size))
            x = np.fft.irfft(spec, 1000)
            out[j] = gain * x / (np.std(x) + 1e-12) * (0.6 + 0.4 * np.abs(np.sin(2 * np.pi * 0.4 * (t + np.arange(1000) / 1000))))
        return out

    def _imu(self, t: float) -> np.ndarray:
        p = (self.effort(t) - 6) / 14
        f_rep = 0.50 - 0.16 * p                                  # repetitions slow down
        ts = t + np.arange(100) / 100
        out = np.empty((N_IMU, 100))
        for i, name in enumerate(IMU_NAMES):
            if name.startswith("acc"):
                out[i] = (9.81 if name.endswith("z") else 0.0) + 0.8 * np.sin(2 * np.pi * f_rep * ts + self.phase[i]) \
                    + (0.15 + 0.35 * p) * self.rng.normal(size=100)
            else:
                out[i] = 1.2 * np.sin(2 * np.pi * f_rep * ts + self.phase[i]) + (0.05 + 0.2 * p) * self.rng.normal(size=100)
        return out

    def _ppg(self, t: float) -> np.ndarray:
        p = (self.effort(t) - 6) / 14
        hr = (80 + 45 * p) / 60
        ts = t + np.arange(200) / 200
        wave = np.sin(2 * np.pi * hr * ts) + 0.35 * np.sin(4 * np.pi * hr * ts + 0.6)
        return 1.4 + 0.2 * wave + 0.01 * self.rng.normal(size=200)

    def chunks(self) -> Iterator[Chunk]:
        for s in range(self.seconds):
            ch = Chunk(self._emg(s), self._imu(s), self._ppg(s), t0=float(s),
                       meta={"source": "hardware_simulator", "true_effort": float(self.effort(s + 0.5))})
            for fault in self.faults:                              # e.g. {"t": (60, 75), "stream": "emg", "channel": 0, "kind": "dropout"}
                if fault["t"][0] <= s < fault["t"][1]:
                    arr = {"emg": ch.emg, "imu": ch.imu, "ppg": ch.ppg}[fault["stream"]]
                    row = arr if arr.ndim == 1 else arr[fault.get("channel", 0)]
                    row[:] = np.nan if fault["kind"] == "dropout" else row * 100
                    ch.meta["fault"] = fault["kind"]
            yield ch


# ----------------------------------------------------------------------------- the streaming pipeline (11.2-11.5)
class LivePipeline:
    """Feeds chunks in, returns one result per completed 10 s window (else None). Never blocks on the language model."""

    def __init__(self, population, zmodel: Optional[ZScoreModel] = None, condition: Optional[Dict[str, float]] = None,
                 calibration_windows: int = 4, ema_alpha: float = 0.5, session_log: Optional[Path] = None,
                 explainer: Optional["ExplanationService"] = None, person_id: str = "P00"):
        self.pop, self.zmodel, self.K, self.alpha = population, zmodel, calibration_windows, ema_alpha
        self.cond = condition or {"load_pct": 35, "direction": "internal", "mvic_dir_N": float("nan")}
        self.explainer, self.person_id = explainer, person_id
        self.session_log = Path(session_log) if session_log else None
        if self.session_log:
            self.session_log.parent.mkdir(parents=True, exist_ok=True)
        self.raw_rows: List[Dict[str, float]] = []
        self.results: List[Dict[str, Any]] = []
        self._buf: List[Chunk] = []
        self._ema = {"generalised": None, "personalised": None}
        self.alerts: List[Dict[str, Any]] = []
        self.latencies_ms: List[float] = []
        self._levels_hit = set()

    # -- calibration flow (11.4)
    @property
    def calibration_state(self) -> str:
        n = len(self.raw_rows)
        if n < N_BASE:
            return f"forming baseline ({n}/{N_BASE} windows)"
        if n < self.K:
            return f"calibrating ({n}/{self.K} windows)"
        return "calibrated"

    def push(self, ch: Chunk) -> Optional[Dict[str, Any]]:
        self._buf.append(ch)
        if len(self._buf) < WINDOW_S // CHUNK_S:
            return None
        t0 = time.perf_counter()
        res = self._process_window(self._buf)
        self._buf = []
        res["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        self.latencies_ms.append(res["latency_ms"])
        self.results.append(res)
        if self.session_log:
            with open(self.session_log, "a", encoding="utf-8") as f:                 # session logging (11.5)
                f.write(json.dumps({k: v for k, v in res.items() if k != "features"}) + "\n")
        if self.explainer is not None:
            self.explainer.submit(res, self.person_id)                                # background, non-blocking (11.6)
        return res

    @staticmethod
    def _valid(arr: np.ndarray, limit: float) -> np.ndarray:
        return np.isfinite(arr) & (np.abs(arr) <= limit)

    def _process_window(self, chunks: List[Chunk]) -> Dict[str, Any]:
        win = len(self.raw_rows)
        z = {"emg": np.concatenate([x.emg for x in chunks], axis=1), "imu": np.concatenate([x.imu for x in chunks], axis=1),
             "ppg": np.concatenate([x.ppg for x in chunks])}
        w: Dict[str, float] = {"t_start": 0.0, "t_end": float(WINDOW_S)}
        for j, m in enumerate(c.MUSCLES):
            w[f"valid_emg_{m}"] = float(self._valid(z["emg"][j], CONTRACT["emg"]["abs_max"]).mean())
        for site in c.IMU_SITES.values():
            idx = [i for i, n in enumerate(IMU_NAMES) if n.split("_", 1)[1].rsplit("_", 1)[0] == site]
            lim = np.array([160.0 if IMU_NAMES[i].startswith("acc") else 35.0 for i in idx])[:, None]
            w[f"valid_imu_{site}"] = float(self._valid(z["imu"][idx], lim).mean())
        w["valid_ppg"] = float(self._valid(z["ppg"], 3.3).mean())
        feats = window_feats(z, IMU_NAMES, w)
        row = {**feats, "win": win, "t_end": chunks[-1].t0 + CHUNK_S}
        self.raw_rows.append(row)

        res: Dict[str, Any] = {"win": win, "t": row["t_end"], "calibration": self.calibration_state,
                               "quality_flags": [k[6:] for k, v in w.items() if k.startswith("valid_") and v < MIN_VALID_FRAC],
                               "generalised": None, "personalised": None, "features": feats}
        if win >= N_BASE - 1:                                  # baseline = mean of the first N_BASE windows
            self._estimate(res)
        for name in ("generalised", "personalised"):
            v = res[name]
            if v is not None:
                self._ema[name] = v if self._ema[name] is None else self.alpha * v + (1 - self.alpha) * self._ema[name]
            res["smoothed_" + name] = None if self._ema[name] is None else round(self._ema[name], 3)
        res["alerts"] = self._alerts(res)
        return res

    # -- model inference (11.3)
    def _frame(self) -> pd.DataFrame:
        F = pd.DataFrame(self.raw_rows)
        sig = [x for x in F.columns if x.startswith(("emg_", "imu_", "tempo_", "trunk_", "hr", "hrv_"))]
        base = F[F.win < N_BASE][sig].mean()
        for s in sig:
            F[s + "_rel"] = (F[s] - base[s]) / (abs(base[s]) + 1e-9)
        F["load_pct"] = self.cond["load_pct"]
        F["dir_int"] = int(self.cond["direction"] == "internal")
        F["mvic_dir_N"] = self.cond["mvic_dir_N"]
        F["target_load_N"] = self.cond["mvic_dir_N"] * self.cond["load_pct"] / 100
        return F

    def _estimate(self, res: Dict[str, Any]) -> None:
        F = self._frame()
        last = F.iloc[[-1]]
        res["generalised"] = float(self.pop.predict(last)[0])
        res["personalised"] = res["generalised"]
        if self.zmodel is not None and len(F) > self.K - 1 and res["win"] >= self.K:
            calib = F.iloc[:self.K]
            zf = ZScoreModel.z(last, calib, self.zmodel.raw_cols)
            res["personalised"] = float(np.clip(self.zmodel.model.predict(zf[self.zmodel.zcols].to_numpy(float))[0], 6, 20))

    def _alerts(self, res: Dict[str, Any]) -> List[str]:
        v = res.get("smoothed_personalised") if res["win"] >= self.K and res.get("smoothed_personalised") is not None \
            else res.get("smoothed_generalised")
        out = []
        if v is not None:
            for level, name in ((WARN_LEVEL, "warning"), (STOP_LEVEL, "stop")):
                if v >= level and name not in self._levels_hit:
                    self._levels_hit.add(name)
                    out.append(name)
                    self.alerts.append({"level": name, "t": res["t"], "value": round(v, 2)})
        return out


# ----------------------------------------------------------------------------- explanation as a background service (11.6)
class ExplanationService:
    """Turns window results into explanations on a worker thread. ``submit`` returns immediately."""

    def __init__(self, router, max_workers: int = 1, usual_rate_per_min: Optional[float] = None):
        self.router = router
        self.usual_rate = usual_rate_per_min          # this person's usual effort build-up; None = fixed thresholds
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="live-explain")
        self.latest: Optional[Dict[str, Any]] = None
        self.history: List[Dict[str, Any]] = []
        self.skipped = 0
        self._prev: Optional[float] = None
        self._lock = threading.Lock()
        self._futures: List[Any] = []

    def submit(self, res: Dict[str, Any], person_id: str) -> None:
        v = res.get("smoothed_personalised") if res.get("smoothed_personalised") is not None else res.get("smoothed_generalised")
        if v is None:
            return
        from ml.rag.session_summary import build_session_summary_from_pipeline
        prev = self._prev if self._prev is not None else v
        self._prev = v
        summary = build_session_summary_from_pipeline(
            person_id=person_id, current_rpe=v, previous_rpe=prev, time_span_seconds=float(WINDOW_S),
            quality_warnings=res["quality_flags"] or None, is_calibrated=res["win"] >= 4,
            model_name="Gradient boosting on muscle, movement and pulse features", usual_rate_per_min=self.usual_rate)
        summary.metadata.update({"elapsed_seconds": res["t"]})
        fire, trigger = self.router.should_trigger_explanation(summary)
        if not fire:
            self.skipped += 1
            return
        self._futures.append(self._pool.submit(self._run, summary, trigger, res["t"]))

    def _run(self, summary, trigger: str, t: float) -> None:
        resp = self.router.explain(summary, force_trigger=True)
        item = {"t": t, "trigger": trigger, "text": resp.answer, "is_llm": resp.is_llm_generated,
                "fallback_reason": resp.fallback_reason, "delay_ms": round(resp.execution_time_ms, 1)}
        with self._lock:
            self.latest = item
            self.history.append(item)

    def wait(self, timeout: float = 30.0) -> None:
        """For tests and the end-to-end run only: the live loop itself never calls this."""
        for f in self._futures:
            f.result(timeout=timeout)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)


def load_models(k: int = 4):
    """(population model, calibrated z-score model) saved by Module 7."""
    import pickle
    with open(c.MODELS / "population_boosting.pkl", "rb") as f:
        pop = pickle.load(f)
    zpath = c.MODELS / f"zscore_live_K{k}.pkl"
    zm = None
    if zpath.exists():
        with open(zpath, "rb") as f:
            zm = pickle.load(f)
    return pop, zm
