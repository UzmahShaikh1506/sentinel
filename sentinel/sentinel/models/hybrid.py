"""
SENTINEL-TS :: The hybrid detector.

Architecture
------------

::

    y_t ──► [ Stage 1: SARIMAX + Fourier ]──► L̂_t  (linear forecast)
                     │
                     ▼
              e_t = y_t − L̂_t              (linear residual)
                     │
                     ▼
            [ Stage 2: nonlinear learner ]──► N̂_t  (residual correction)
                     │
                     ▼
              ŷ_t = L̂_t + N̂_t              (hybrid forecast)
                     │
                     ▼
              r_t = y_t − ŷ_t               (hybrid residual)
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
   robust z of r_t      Isolation Forest over
   (train-calibrated)   residual-shape features
         └───────────┬────────────┘
                     ▼
        weighted sigma-equivalent score ──► debounce ──► alert

Why this decomposition
----------------------
The premise, following Zhang's hybrid methodology, is that a series is the sum
of a linear component a well-specified ARMA can capture and a nonlinear one it
structurally cannot. Fitting one model to both forces a compromise. Fitting
them in sequence lets each do what it is good at: SARIMAX extracts the
autocorrelation and the daily cycle; the learner picks up what survives --
burst clustering, asymmetric recovery after a spike, weekday-specific
behaviour that is not a pure harmonic.

Why two detectors on top
------------------------
The two failure modes are different and largely independent. A residual
threshold is sharp on magnitude but blind to shape: a slow ramp that never
exceeds sigma stays invisible. Isolation Forest is sensitive to shape but
indifferent to magnitude and produces no forecast. Combining them in
sigma-equivalent units - rather than by rank, which would destroy the
interpretable threshold - keeps one calibrated scale while covering both.

Why the debounce
----------------
Single-bucket exceedances are overwhelmingly noise. Requiring the score to
stay above threshold for a short dwell converts a large share of false alarms
into nothing at all, at the cost of one bucket of detection latency. Both
sides of that trade are measured and reported in the leaderboard.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from ..preprocessing import robust_scale
from .base import BaseDetector, calendar_features
from .classical import SarimaxFourier
from .ml import ResidualLearner


def _residual_shape_features(residual: pd.Series) -> np.ndarray:
    """Causal descriptors of residual *behaviour* rather than magnitude.

    Captures local volatility, momentum and persistence, which is what lets
    the forest see a developing ramp before any single residual is large.
    """
    shifted = residual.shift(1)
    frame = pd.DataFrame(
        {
            "r_lag1": shifted,
            "r_lag2": residual.shift(2),
            "r_lag3": residual.shift(3),
            "roll_mean_8": shifted.rolling(8, min_periods=2).mean(),
            "roll_std_8": shifted.rolling(8, min_periods=2).std(),
            "roll_mean_32": shifted.rolling(32, min_periods=4).mean(),
            "roll_std_32": shifted.rolling(32, min_periods=4).std(),
            "abs_roll_mean_8": shifted.abs().rolling(8, min_periods=2).mean(),
            "slope_8": shifted - shifted.shift(8),
        },
        index=residual.index,
    )
    cal = calendar_features(residual.index)
    for i, label in enumerate(["tod_sin", "tod_cos", "dow_sin", "dow_cos", "wknd"]):
        frame[label] = cal[:, i]
    return np.nan_to_num(frame.bfill().ffill().to_numpy(dtype="float64"), nan=0.0)


def debounce(flags: np.ndarray, min_consecutive: int) -> np.ndarray:
    """Suppress alert runs shorter than ``min_consecutive`` buckets."""
    if min_consecutive <= 1:
        return flags
    out = np.zeros_like(flags, dtype=bool)
    run_start: Optional[int] = None
    for i, flag in enumerate(flags):
        if flag and run_start is None:
            run_start = i
        elif not flag and run_start is not None:
            if i - run_start >= min_consecutive:
                out[run_start:i] = True
            run_start = None
    if run_start is not None and len(flags) - run_start >= min_consecutive:
        out[run_start:] = True
    return out


class HybridDetector(BaseDetector):
    name = "hybrid"
    display_name = "Hybrid SARIMAX + GBR"
    family = "hybrid"
    forecasts = True
    description = (
        "Two-stage forecast - SARIMAX with Fourier seasonality, corrected by a "
        "nonlinear learner trained on its residuals - feeding a dual detector "
        "that fuses a train-calibrated robust residual z-score with an "
        "Isolation Forest over residual-shape features, then debounces."
    )
    rationale = (
        "Each stage exists because a measured weakness of the previous one "
        "demanded it. SARIMAX leaves structured residuals; the nonlinear stage "
        "absorbs them. A magnitude threshold alone misses slow ramps; the "
        "residual-shape forest catches them. Raw thresholding alarms "
        "constantly; the debounce trades one bucket of latency for a large "
        "reduction in false pages."
    )

    #: Weights for the sigma-equivalent fusion. Magnitude dominates because it
    #: is the better-calibrated and more explainable of the two signals; the
    #: shape term is a sensitivity booster, not a co-equal vote.
    W_MAGNITUDE = 0.70
    W_SHAPE = 0.30

    #: Ceiling on the shape channel, in sigma-equivalent units. This enforces
    #: the stated design rather than merely describing it: with a weight of
    #: 0.30 and a cap of 6.0, the shape term contributes at most 1.8, which is
    #: below any sane threshold. Shape can therefore tip a borderline
    #: magnitude reading over the line, but can never raise an alert on its
    #: own. Without the cap, a training window that happens to be very regular
    #: gives the forest a razor-thin manifold, every hold-out point scores as
    #: strange, and the hybrid alarms continuously.
    SHAPE_CAP = 6.0

    #: Clamp on the nonlinear correction, in units of the training linear
    #: residual scale.
    #:
    #: This exists because of a failure found during cross-dataset evaluation,
    #: and it is the single most important detail in this class. An unclamped
    #: residual learner predicts e_t from its own recent lags. During a
    #: *sustained* anomaly the residuals are large but highly persistent, so
    #: the learner forecasts them accurately, the correction cancels them, the
    #: hybrid residual collapses towards zero -- and the detector goes quiet
    #: precisely when the incident is happening. On the ELB request-count
    #: service the first version of this model scored 0.00 segment recall for
    #: exactly that reason while every other model caught both windows.
    #:
    #: The fix follows from what stage two is *for*: tightening ordinary
    #: residuals, not tracking excursions. Lagged inputs are winsorised and the
    #: correction is clamped at this many training sigmas, so during a genuine
    #: excursion the correction saturates and the hybrid degrades gracefully to
    #: its linear stage, which detects the event. In normal operation the
    #: clamp almost never binds and the variance reduction is retained.
    RESIDUAL_CLAMP = 2.0

    def __init__(self, *args, residual_kind: str = "gbr", **kwargs):
        super().__init__(*args, **kwargs)
        self.residual_kind = residual_kind

    # -- fitting -----------------------------------------------------------

    def fit(self, train: pd.Series) -> "HybridDetector":
        self._train = train
        self._train_len = len(train)
        self._min_consecutive = int(self.options.get("min_consecutive", 2))

        # ---- Stage 1: linear -------------------------------------------
        self._linear = SarimaxFourier(
            period=self.period,
            threshold_sigma=self.threshold_sigma,
            fourier_k=self.options.get("fourier_k", 4),
            suggested_order=self.options.get("suggested_order"),
        )
        self._linear.fit(train)

        linear_fitted = np.asarray(
            self._linear._result.fittedvalues, dtype="float64"
        )
        linear_resid = train.to_numpy(dtype="float64") - linear_fitted

        burn = max(self.period // 4, 8)
        self._burn = burn

        # Scale of the linear residuals, used to bound stage two (see
        # RESIDUAL_CLAMP). Taken from training residuals past the burn-in and
        # measured robustly so a single excursion cannot widen the clamp.
        self._lin_centre = float(np.median(linear_resid[burn:]))
        self._lin_scale = max(robust_scale(linear_resid[burn:]), 1e-9)
        self._clamp = self.RESIDUAL_CLAMP * self._lin_scale

        # ---- Stage 2: nonlinear ----------------------------------------
        self._learner = ResidualLearner(
            kind=self.residual_kind, lags=int(self.options.get("residual_lags", 12))
        )
        bounded_resid = self._winsorise(linear_resid)
        self._learner.fit(bounded_resid, train.index)

        correction_train = self._correct(bounded_resid, train.index)
        hybrid_resid_train = linear_resid - correction_train

        # Calibrate the magnitude channel on training residuals only.
        calib = hybrid_resid_train[burn:]
        self._calibrate_residuals(calib)
        self._resid_series_train = pd.Series(hybrid_resid_train, index=train.index)

        # ---- Shape channel ---------------------------------------------
        shape_train = _residual_shape_features(self._resid_series_train)
        self._shape_scaler = StandardScaler().fit(shape_train)
        self._forest = IsolationForest(
            n_estimators=250,
            contamination=float(self.options.get("contamination", 0.03)),
            random_state=RANDOM_SEED,
            n_jobs=1,
        ).fit(self._shape_scaler.transform(shape_train))

        forest_train = -self._forest.decision_function(
            self._shape_scaler.transform(shape_train)
        )
        self._forest_centre = float(np.median(forest_train[burn:]))
        self._forest_scale = max(robust_scale(forest_train[burn:]), 1e-9)

        # ---- Reporting ---------------------------------------------------
        var_linear = float(np.var(linear_resid[burn:]))
        var_hybrid = float(np.var(hybrid_resid_train[burn:]))
        improvement = (
            100.0 * (1.0 - var_hybrid / var_linear) if var_linear > 1e-12 else 0.0
        )

        self.params = {
            "stage_1": "SARIMAX + Fourier",
            "stage_1_order": self._linear.params.get("order"),
            "stage_1_fourier_harmonics": self._linear.params.get("fourier_harmonics"),
            "stage_2": self._learner.describe()["label"],
            "stage_2_lags": self._learner.lags,
            "fusion_weights": {
                "magnitude": self.W_MAGNITUDE,
                "shape": self.W_SHAPE,
            },
            "shape_channel_cap_sigma": self.SHAPE_CAP,
            "residual_correction_clamp_sigma": self.RESIDUAL_CLAMP,
            "debounce_buckets": self._min_consecutive,
            "threshold_sigma": self.threshold_sigma,
        }
        self.diagnostics = {
            "residual_learner": self._learner.describe(),
            "linear_residual_variance": round(var_linear, 5),
            "hybrid_residual_variance": round(var_hybrid, 5),
            "residual_variance_reduction_pct": round(improvement, 2),
            "correction_clamp_absolute": round(float(self._clamp), 5),
            "correction_clamp_hit_rate_train": round(
                float(np.mean(np.abs(correction_train) >= self._clamp * 0.999)), 5
            ),
            "stage_1_order_search": self._linear.diagnostics.get("order_search"),
            "interpretation": (
                f"The nonlinear stage removed {improvement:.1f}% of the variance "
                "the linear model left in its residuals. A figure near zero "
                "would be evidence that SARIMAX had already extracted "
                "everything predictable and that the hybrid is unnecessary "
                "complexity for this series - which is a legitimate finding, "
                "not a failure."
            ),
        }
        return self

    # -- stage-two bounding -------------------------------------------------

    def _winsorise(self, residual: np.ndarray) -> np.ndarray:
        """Bound residuals before they are fed to the nonlinear learner."""
        lo = self._lin_centre - self._clamp
        hi = self._lin_centre + self._clamp
        return np.clip(np.asarray(residual, dtype="float64"), lo, hi)

    def _correct(self, bounded_residual: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
        """Nonlinear correction, clamped so it cannot absorb an excursion."""
        raw = self._learner.predict(bounded_residual, index)
        return np.clip(raw, -self._clamp, self._clamp)

    # -- prediction and scoring -------------------------------------------

    def predict(self, test: pd.Series) -> Optional[np.ndarray]:
        linear_pred = self._linear.predict(test)
        linear_resid_test = test.to_numpy(dtype="float64") - linear_pred

        # Warm the lag window with training residuals so the first test points
        # are corrected with a full history rather than zeros.
        linear_fitted_train = np.asarray(
            self._linear._result.fittedvalues, dtype="float64"
        )
        train_resid = self._train.to_numpy(dtype="float64") - linear_fitted_train
        combined_resid = np.concatenate([train_resid, linear_resid_test])
        combined_index = self._train.index.append(test.index)

        correction = self._correct(
            self._winsorise(combined_resid), combined_index
        )[self._train_len :]
        self._last_correction = correction
        self._last_linear_pred = linear_pred
        return linear_pred + correction

    def score(self, test: pd.Series, predictions: Optional[np.ndarray]) -> np.ndarray:
        residual = test.to_numpy(dtype="float64") - np.asarray(
            predictions, dtype="float64"
        )

        # Channel A: magnitude, in sigma units against the training residual.
        z_magnitude = np.abs(
            (residual - self._resid_centre) / max(self._resid_scale, 1e-9)
        )

        # Channel B: residual shape, rescaled to the same sigma-equivalent axis.
        combined = pd.concat(
            [self._resid_series_train, pd.Series(residual, index=test.index)]
        )
        shape = _residual_shape_features(combined)[self._train_len :]
        forest_raw = -self._forest.decision_function(
            self._shape_scaler.transform(shape)
        )
        z_shape = np.clip(
            (forest_raw - self._forest_centre) / self._forest_scale,
            0.0,
            self.SHAPE_CAP,
        )

        self._channels = {
            "magnitude": z_magnitude,
            "shape": z_shape,
        }
        return self.W_MAGNITUDE * z_magnitude + self.W_SHAPE * z_shape

    def flag(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        raw = np.asarray(scores, dtype="float64") > threshold
        return debounce(raw, self._min_consecutive)

    def forecast_future(self, horizon: int) -> Optional[Dict[str, object]]:
        """Forward forecast from the linear stage.

        The nonlinear correction is deliberately not extrapolated. It is a
        function of recent residuals, which do not exist beyond the data; using
        its own predictions as inputs would compound error with no accuracy
        gain and would hand the operator a falsely confident curve.
        """
        base = self._linear.forecast_future(horizon)
        if base is None:
            return None
        base["method"] = (
            "Linear stage multi-step forecast with 95% prediction interval. "
            "The nonlinear residual correction is not extrapolated beyond the "
            "observed data by design."
        )
        return base


class HybridNeuralDetector(HybridDetector):
    """Ablation: identical pipeline, neural residual stage.

    Present so the comparison can answer a question the leaderboard would
    otherwise beg - whether the hybrid's advantage comes from the two-stage
    architecture itself or from the particular learner bolted into stage two.
    """

    name = "hybrid_neural"
    display_name = "Hybrid SARIMAX + Neural"
    family = "hybrid"
    description = (
        "The same two-stage architecture with a feed-forward neural network "
        "(or an LSTM where PyTorch is installed) in place of gradient boosting "
        "at the residual-correction stage."
    )
    rationale = (
        "An ablation of the residual learner. If this and the boosted variant "
        "score alike, the gain is architectural; if they diverge sharply, the "
        "choice of learner is doing the work and should be tuned per series."
    )

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("residual_kind", "lstm")
        super().__init__(*args, **kwargs)
