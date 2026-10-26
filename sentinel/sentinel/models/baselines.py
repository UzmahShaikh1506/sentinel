"""
SENTINEL-TS :: Baselines.

Two deliberately simple models exist so that every sophisticated result has
something honest to beat.

**Seasonal naive** predicts today's value from yesterday's at the same
time-of-day. It is the reference forecaster the MASE metric is scaled against,
and on strongly diurnal telemetry it is far harder to beat than it looks.

**Rolling z-score** is what most production monitoring actually ships: a
trailing window, a robust centre and spread, and a sigma threshold. It carries
no model of seasonality at all, which is exactly the weakness the project sets
out to demonstrate.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..preprocessing import rolling_robust_zscore, seasonal_profile
from .base import BaseDetector


class SeasonalNaive(BaseDetector):
    name = "seasonal_naive"
    display_name = "Seasonal Naive"
    family = "baseline"
    forecasts = True
    description = (
        "Predicts each value from the observation exactly one seasonal period "
        "earlier, then alerts on a robust z-score of the residual."
    )
    rationale = (
        "The reference point for the whole comparison. MASE is defined against "
        "this model, so any method with MASE >= 1 has not earned its "
        "complexity. On diurnal infrastructure telemetry it is a genuinely "
        "strong baseline, not a straw man."
    )

    def fit(self, train: pd.Series) -> "SeasonalNaive":
        self._train = train
        self._profile = seasonal_profile(train, self.period)
        self._train_len = len(train)

        m = self.period
        values = train.to_numpy(dtype="float64")
        if len(values) > m:
            residual = values[m:] - values[:-m]
        else:  # pragma: no cover - very short series
            residual = values - float(np.median(values))
        self._calibrate_residuals(residual)

        self.params = {"seasonal_period": m, "n_train": int(len(train))}
        return self

    def predict(self, test: pd.Series) -> Optional[np.ndarray]:
        """One-step-ahead: the value one seasonal period before each point.

        For early test points that lag still falls inside the training set, so
        the two series are concatenated and the lag taken over the join. No
        future information is used: the lag is always backwards.
        """
        m = self.period
        combined = pd.concat([self._train, test])
        lagged = combined.shift(m)
        out = lagged.iloc[self._train_len :].to_numpy(dtype="float64")

        if np.isnan(out).any() and not self._profile.empty:
            positions = (np.arange(self._train_len, len(combined))) % m
            fallback = self._profile.reindex(positions).to_numpy(dtype="float64")
            out = np.where(np.isnan(out), fallback, out)
        return np.nan_to_num(out, nan=float(self._train.median()))

    def forecast_future(self, horizon: int) -> Optional[Dict[str, object]]:
        if self._profile.empty:
            return None
        start = self._train_len % self.period
        positions = (np.arange(horizon) + start) % self.period
        values = self._profile.reindex(positions).to_numpy(dtype="float64")
        return {
            "values": [round(float(v), 4) for v in np.nan_to_num(values)],
            "method": "seasonal profile repetition",
        }


class RollingZScore(BaseDetector):
    name = "rolling_zscore"
    display_name = "Rolling Z-Score (3-sigma)"
    family = "statistical"
    forecasts = False
    description = (
        "Classical statistical process control: a trailing window supplies a "
        "robust median and MAD-scaled spread, and any point beyond the sigma "
        "threshold is alerted."
    )
    rationale = (
        "The method most monitoring stacks ship by default. It has no notion of "
        "time-of-day, so it is included to quantify the cost of ignoring "
        "seasonality: it tends to alert every night when load legitimately "
        "collapses, and to miss daytime incidents hidden inside the peak."
    )

    def fit(self, train: pd.Series) -> "RollingZScore":
        self._train = train
        self._window = int(self.options.get("rolling_window", min(self.period, 96)))
        self._train_len = len(train)
        self.params = {
            "window_buckets": self._window,
            "scale_estimator": "median absolute deviation x 1.4826",
            "threshold_sigma": self.threshold_sigma,
        }
        return self

    def score(self, test: pd.Series, predictions: Optional[np.ndarray]) -> np.ndarray:
        """Score from a causal rolling window spanning the train/test join.

        The window is warmed up on training data so the first test points are
        not scored against an almost-empty window.
        """
        combined = pd.concat([self._train, test])
        z = rolling_robust_zscore(combined, window=self._window)
        return np.abs(z.iloc[self._train_len :].to_numpy(dtype="float64"))

    def threshold_for(self, scores: np.ndarray) -> float:
        return float(self.threshold_sigma)
