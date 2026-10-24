"""Module 7 core: population model + calibration functions for a new person.

    pop = fit_population(F)                          # boosting on fused rel-sensors + time + condition
    pred = calibrate(pop, F_calib, method="bias")    # F_calib: the new person's calibration windows (with rpe)
    y_hat = pred(F_new)                              # predictions for that person's later windows

Methods: "none" (generalised), "bias", "linear", "weighted" (retrain population + calibration windows, weight w).
"""

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.dataset_pipeline import common as c
from ml.dataset_pipeline.m5_baselines import feature_sets

COND = ["t_end", "load_pct", "dir_int", "target_load_N", "mvic_dir_N"]


def feature_columns(F):
    return feature_sets(F, True)["FUSED"] + COND


def new_boost():
    return make_pipeline(SimpleImputer(strategy="median"),
                         HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=250,
                                                       min_samples_leaf=20, random_state=0))


class Population:
    def __init__(self, cols, X, y, model):
        self.cols, self.X, self.y, self.model = cols, X, y, model

    def predict(self, F):
        return np.clip(self.model.predict(F[self.cols].to_numpy(float)), 6, 20)


def fit_population(F):
    cols = feature_columns(F)
    X, y = F[cols].to_numpy(float), F.rpe.to_numpy(float)
    return Population(cols, X, y, new_boost().fit(X, y))


class ZScoreModel:
    """Module 7.1 (best method). Features of a recording are z-scored with that recording's own calibration windows,
    then fed to a population boosting model trained on z-scored features of everyone else."""

    def __init__(self, raw_cols, model):
        self.raw_cols, self.model = raw_cols, model
        self.zcols = [x + "_z" for x in raw_cols] + COND

    @staticmethod
    def z(df, calib, raw_cols):
        mu, sd = calib[raw_cols].mean(), calib[raw_cols].std().fillna(0)
        out = df.copy()
        for k in raw_cols:
            out[k + "_z"] = ((df[k] - mu[k]) / (sd[k] + 1e-6 * (abs(mu[k]) + 1e-9))).clip(-10, 10)
        return out

    def calibrated_predictor(self, calib):
        """calib: first windows of ONE recording of a new person (needs the sensor features, not rpe). Returns predict(F)."""
        return lambda F: np.clip(self.model.predict(self.z(F, calib, self.raw_cols)[self.zcols].to_numpy(float)), 6, 20)


def calibrate(pop, calib, method="bias", weight=10.0, alpha=10.0):
    """Return predict(F) personalised with the person's own calibration windows (never their later windows)."""
    yc = calib.rpe.to_numpy(float)
    pc = pop.predict(calib)
    if method == "none" or len(calib) < 3:
        return pop.predict
    if method == "bias":
        b = np.mean(yc - pc)
        return lambda F: np.clip(pop.predict(F) + b, 6, 20)
    if method == "linear":                           # residual ~ ridge([population prediction, elapsed time])
        Z = lambda F: np.c_[pop.predict(F), F.t_end.to_numpy(float)]
        m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(Z(calib), yc - pc)
        return lambda F: np.clip(pop.predict(F) + m.predict(Z(F)), 6, 20)
    if method == "weighted":                         # fine-tune: population + calibration windows with extra weight
        X = np.vstack([pop.X, calib[pop.cols].to_numpy(float)])
        y = np.r_[pop.y, yc]
        w = np.r_[np.ones(len(pop.y)), np.full(len(yc), weight)]
        m = new_boost()
        m.fit(X, y, histgradientboostingregressor__sample_weight=w)
        return lambda F: np.clip(m.predict(F[pop.cols].to_numpy(float)), 6, 20)
    raise ValueError(method)
