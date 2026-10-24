"""Unit tests for ml.dataset_pipeline (Modules 1-7). Synthetic inputs only: no dataset needed."""

import numpy as np
import pandas as pd
import pytest

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m2_eda import partial_spearman
from ml.dataset_pipeline.m3_clean import flat_runs, repair
from ml.dataset_pipeline.m4_features import MIN_VALID_FRAC, add_relative, window_feats
from ml.dataset_pipeline.m5_baselines import metrics, time_split
from ml.dataset_pipeline.personalise import ZScoreModel


def test_flat_runs_flags_only_long_runs():
    x = np.r_[np.arange(5.0), np.full(10, 3.0), np.arange(5.0)]
    assert flat_runs(x, 8).sum() == 10
    assert not flat_runs(x, 11).any()


def test_repair_interpolates_short_gaps_only():
    x = np.arange(100, dtype=float)
    bad = np.zeros(100, bool)
    bad[10:13] = True       # short gap
    bad[50:70] = True       # long gap
    y, missing = repair(x, bad, max_gap=5)
    assert np.allclose(y[10:13], x[10:13]) and not missing[10:13].any()
    assert missing[50:70].all() and np.isnan(y[50:70]).all()


def test_task_folder_found_by_pattern(tmp_path, monkeypatch):
    root = tmp_path / "EMG, IMU, and PPG data"
    for name in ("30–40% external rotation", "40-50_ internal rotation", "MVIC force external rotation first"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(c, "DATA", tmp_path)
    assert c.task_folder("task4_35e").name.startswith("30")
    assert c.task_folder("task2_45i").name.startswith("40")
    with pytest.raises(FileNotFoundError):
        c.task_folder("task6_55e")


def test_time_split_is_ordered_and_leak_free():
    rows = [dict(sid=s, task="t", win=w) for s in (1, 2) for w in range(8)]
    F = pd.DataFrame(rows)
    cal, test = time_split(F)
    assert not set(cal) & set(test)
    for s in (1, 2):
        w_cal = F.loc[cal][F.loc[cal].sid == s].win
        w_test = F.loc[test][F.loc[test].sid == s].win
        assert w_cal.max() < w_test.min()


def test_metrics_within_one_and_mae():
    y = np.array([10.0, 12.0, 14.0, 16.0])
    p = np.array([10.5, 12.0, 17.0, 16.0])
    m = metrics(y, p)
    assert m["MAE"] == pytest.approx(0.875)
    assert m["within1"] == pytest.approx(0.75)


def test_partial_spearman_removes_shared_time_trend():
    rng = np.random.default_rng(0)
    t = np.arange(60.0)
    x, y = t + rng.normal(0, 3, 60), t + rng.normal(0, 3, 60)
    from scipy.stats import spearmanr
    assert spearmanr(x, y)[0] > 0.9
    assert abs(partial_spearman(x, y, t)) < 0.4


def test_window_feats_are_nan_when_channel_mostly_invalid():
    w = {"t_start": 0.0, "t_end": 10.0, "valid_ppg": 0.0, "valid_imu_sternum": 0.0}
    w.update({f"valid_emg_{m}": MIN_VALID_FRAC - 0.01 for m in c.MUSCLES})
    w.update({f"valid_imu_{t}": 0.0 for t in c.IMU_SITES.values()})
    r = window_feats({"emg": None, "imu": None, "ppg": None}, [], w)
    assert all(np.isnan(v) for k, v in r.items() if k != "ppg_beats")


def test_baseline_relative_uses_only_first_windows():
    F = pd.DataFrame({"sid": 1, "task": "t", "win": range(5), "emg_x_rms": [1.0, 1.0, 2.0, 3.0, 4.0]})
    out = add_relative(F, ["emg_x_rms"])
    assert out.emg_x_rms_rel.tolist() == pytest.approx([0.0, 0.0, 1.0, 2.0, 3.0])


def test_zscore_uses_calibration_statistics_only():
    calib = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    new = pd.DataFrame({"f": [2.0, 100.0]})
    z = ZScoreModel.z(new, calib, ["f"])
    assert z.f_z.iloc[0] == pytest.approx(0.0, abs=1e-3)
    assert z.f_z.iloc[1] == pytest.approx(10.0)          # clipped at 10; later windows never change mu/sd
