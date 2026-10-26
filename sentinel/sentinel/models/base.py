"""
SENTINEL-TS :: Model interface.

Every detector in the zoo implements the same contract so the comparison
harness can treat them interchangeably. The contract separates three ideas
that are often conflated:

``predict``
    A one-step-ahead point forecast for each test timestamp, using only
    information available before that timestamp. Pure detectors (Isolation
    Forest, rolling z-score) return ``None`` here and are simply not scored on
    forecast accuracy.

``score``
    A continuous anomaly score per test timestamp. Higher means stranger.
    Scores are *not* comparable across models in raw units, so the harness
    normalises them before display.

``flags``
    The boolean alert decision produced by applying the model's own threshold
    rule to its scores.

Keeping the three apart is what makes the leaderboard honest: a model can
forecast well and alert badly, or vice versa, and the table will say so.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..preprocessing import robust_scale


@dataclass
class ModelOutput:
    """Everything one fitted model contributes to the comparison."""

    name: str
    display_name: str
    family: str
    description: str
    rationale: str
    scores: np.ndarray
    flags: np.ndarray
    threshold: float
    predictions: Optional[np.ndarray] = None
    lower: Optional[np.ndarray] = None
    upper: Optional[np.ndarray] = None
    params: Dict[str, object] = field(default_factory=dict)
    fit_seconds: float = 0.0
    diagnostics: Dict[str, object] = field(default_factory=dict)
    forecast_future: Optional[Dict[str, object]] = None
    error: Optional[str] = None


class BaseDetector:
    """Common behaviour for every detector."""

    name = "base"
    display_name = "Base"
    family = "baseline"
    description = ""
    rationale = ""
    #: True when the model produces a point forecast worth scoring.
    forecasts = False

    def __init__(self, period: int = 96, threshold_sigma: float = 3.5, **kwargs):
        self.period = int(period)
        self.threshold_sigma = float(threshold_sigma)
        self.options = kwargs
        self.params: Dict[str, object] = {}
        self.diagnostics: Dict[str, object] = {}
        self._fit_seconds = 0.0

    # -- to be implemented by subclasses ----------------------------------

    def fit(self, train: pd.Series) -> "BaseDetector":  # pragma: no cover
        raise NotImplementedError

    def predict(self, test: pd.Series) -> Optional[np.ndarray]:
        """One-step-ahead point forecasts over ``test``; ``None`` if N/A."""
        return None

    def score(self, test: pd.Series, predictions: Optional[np.ndarray]) -> np.ndarray:
        """Default scoring: rolling robust z of the forecast residual."""
        if predictions is None:
            raise NotImplementedError(
                f"{self.name} must override score() when it does not forecast."
            )
        residual = test.to_numpy(dtype="float64") - np.asarray(
            predictions, dtype="float64"
        )
        return self._residual_score(residual)

    def forecast_future(self, horizon: int) -> Optional[Dict[str, object]]:
        """Forward forecast past the end of the data; ``None`` if unsupported."""
        return None

    # -- shared helpers ----------------------------------------------------

    def _residual_score(self, residual: np.ndarray) -> np.ndarray:
        """Convert residuals into a robust, scale-free anomaly score.

        The location and scale come from the *training* residuals where
        available, so the score does not quietly calibrate itself on the
        incidents it is meant to find.
        """
        residual = np.asarray(residual, dtype="float64")
        centre = float(getattr(self, "_resid_centre", np.nanmedian(residual)))
        scale = float(getattr(self, "_resid_scale", robust_scale(residual)))
        if not np.isfinite(scale) or scale <= 1e-12:
            scale = max(robust_scale(residual), 1e-9)
        return np.abs((residual - centre) / scale)

    def _calibrate_residuals(self, residual: np.ndarray) -> None:
        """Store training-residual location and scale for later scoring."""
        residual = np.asarray(residual, dtype="float64")
        residual = residual[np.isfinite(residual)]
        if residual.size:
            self._resid_centre = float(np.median(residual))
            self._resid_scale = float(robust_scale(residual))
        else:  # pragma: no cover
            self._resid_centre, self._resid_scale = 0.0, 1.0

    def threshold_for(self, scores: np.ndarray) -> float:
        return float(self.threshold_sigma)

    def flag(self, scores: np.ndarray, threshold: float) -> np.ndarray:
        return np.asarray(scores, dtype="float64") > threshold

    # -- orchestration -----------------------------------------------------

    def run(self, train: pd.Series, test: pd.Series) -> ModelOutput:
        """Fit on train, evaluate on test, and package the result.

        A model that fails numerically returns a populated ``ModelOutput``
        carrying the error rather than raising, so one unstable model cannot
        take down a comparison run.
        """
        start = time.perf_counter()
        try:
            self.fit(train)
            predictions = self.predict(test)
            scores = np.asarray(self.score(test, predictions), dtype="float64")
            # A non-finite score means a degenerate scale (a constant window
            # followed by a jump), which is a genuine detection, not noise.
            # Clipping to a large multiple of the threshold keeps the alert
            # while keeping the axis readable; zeroing it would silently
            # discard a real event.
            cap = 100.0 * max(self.threshold_sigma, 1.0)
            scores = np.nan_to_num(scores, nan=0.0, posinf=cap, neginf=0.0)
            scores = np.clip(scores, 0.0, cap)
            threshold = self.threshold_for(scores)
            flags = self.flag(scores, threshold)
            elapsed = time.perf_counter() - start

            lower = upper = None
            if predictions is not None:
                band = threshold * float(getattr(self, "_resid_scale", 1.0))
                centre = float(getattr(self, "_resid_centre", 0.0))
                lower = np.asarray(predictions) + centre - band
                upper = np.asarray(predictions) + centre + band

            return ModelOutput(
                name=self.name,
                display_name=self.display_name,
                family=self.family,
                description=self.description,
                rationale=self.rationale,
                scores=scores,
                flags=flags,
                threshold=float(threshold),
                predictions=(
                    np.asarray(predictions, dtype="float64")
                    if predictions is not None
                    else None
                ),
                lower=lower,
                upper=upper,
                params=dict(self.params),
                fit_seconds=round(elapsed, 3),
                diagnostics=dict(self.diagnostics),
                forecast_future=self.forecast_future(
                    int(self.options.get("horizon", 96))
                ),
            )
        except Exception as exc:  # pragma: no cover - resilience path
            elapsed = time.perf_counter() - start
            n = len(test)
            return ModelOutput(
                name=self.name,
                display_name=self.display_name,
                family=self.family,
                description=self.description,
                rationale=self.rationale,
                scores=np.zeros(n),
                flags=np.zeros(n, dtype=bool),
                threshold=float(self.threshold_sigma),
                fit_seconds=round(elapsed, 3),
                error=f"{type(exc).__name__}: {exc}",
            )


# --------------------------------------------------------------------------
# Shared feature construction
# --------------------------------------------------------------------------


def fourier_terms(
    index: pd.DatetimeIndex,
    period_hours: float = 24.0,
    k: int = 4,
    origin: Optional[pd.Timestamp] = None,
) -> np.ndarray:
    """Harmonic regressors encoding a cycle of ``period_hours``.

    A seasonal ARIMA with m = 96 would need 96 seasonal lags and is not
    estimable at this sample size. Projecting the cycle onto 2k Fourier terms
    captures the same shape with 8 parameters instead, which is the standard
    remedy for long seasonal periods.

    Time is measured from ``origin`` (default: the first timestamp) rather than
    from the Unix epoch. This is not cosmetic: epoch-based arguments run to
    ~4 x 10^5 radians at k = 4, where double-precision sine loses enough
    significant digits to visibly degrade the regressors and, downstream, the
    fitted model.
    """
    if origin is None:
        origin = index[0]
    elapsed = (index - origin).total_seconds().to_numpy(dtype="float64")
    t = elapsed / (period_hours * 3600.0)
    cols = []
    for i in range(1, k + 1):
        cols.append(np.sin(2 * np.pi * i * t))
        cols.append(np.cos(2 * np.pi * i * t))
    return np.column_stack(cols)


def calendar_features(index: pd.DatetimeIndex) -> np.ndarray:
    """Cyclical time-of-day and day-of-week encodings plus a weekend flag."""
    tod = (index.hour * 60 + index.minute) / 1440.0
    dow = index.dayofweek / 7.0
    return np.column_stack(
        [
            np.sin(2 * np.pi * tod),
            np.cos(2 * np.pi * tod),
            np.sin(2 * np.pi * dow),
            np.cos(2 * np.pi * dow),
            (index.dayofweek >= 5).astype("float64"),
        ]
    )


def normalise_scores(scores: np.ndarray) -> np.ndarray:
    """Map an arbitrary score onto [0, 1] for cross-model display.

    Percentile ranking is used rather than min-max because a single extreme
    score would otherwise compress every other model's range to nothing.
    """
    arr = np.asarray(scores, dtype="float64")
    if arr.size == 0:
        return arr
    order = arr.argsort().argsort().astype("float64")
    return order / max(arr.size - 1, 1)
