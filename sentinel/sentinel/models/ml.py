"""
SENTINEL-TS :: Machine-learning components.

``build_window_features``
    Shared causal feature map. Every column at index *t* is computable from
    observations strictly before *t*, which is what makes the same code valid
    for both offline evaluation and online scoring.

``IsolationForestDetector``
    Unsupervised, model-free detection. Isolates points by random axis-aligned
    splits; anomalies need fewer splits and so sit at shallower average depth.
    It makes no assumption of normality, linearity or stationarity, which is
    its advantage over the residual-threshold family -- and it produces no
    forecast at all, which is its cost.

``ResidualLearner``
    The nonlinear half of the hybrid. Learns what the linear model's residuals
    still have left to say, from their own recent history plus calendar
    context. Gradient boosting is the default; a feed-forward neural network
    over the same lag window is available for ablation, and an LSTM path
    activates automatically if PyTorch is installed.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ..config import RANDOM_SEED
from .base import BaseDetector, calendar_features

try:  # pragma: no cover - exercised only where torch is installed
    import torch
    import torch.nn as nn

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False


# --------------------------------------------------------------------------
# Feature construction
# --------------------------------------------------------------------------

DEFAULT_LAGS: Tuple[int, ...] = (1, 2, 3, 4, 8, 12, 24, 96)


def build_window_features(
    series: pd.Series,
    lags: Tuple[int, ...] = DEFAULT_LAGS,
    roll_windows: Tuple[int, ...] = (4, 12, 48),
) -> Tuple[np.ndarray, List[str]]:
    """Causal feature matrix aligned to ``series``.

    Every feature is built from ``shift(>= 1)``, so row *t* contains nothing
    observed at or after *t*. Leading rows that cannot be filled are
    back-filled; they sit inside the training window and are a negligible
    fraction of it.
    """
    frame = pd.DataFrame(index=series.index)
    names: List[str] = []

    for lag in lags:
        if lag < len(series):
            frame[f"lag_{lag}"] = series.shift(lag)
            names.append(f"lag_{lag}")

    for window in roll_windows:
        if window < len(series):
            shifted = series.shift(1)
            frame[f"rmean_{window}"] = shifted.rolling(window, min_periods=2).mean()
            frame[f"rstd_{window}"] = shifted.rolling(window, min_periods=2).std()
            names += [f"rmean_{window}", f"rstd_{window}"]

    frame["diff_1"] = series.diff().shift(1)
    frame["diff_2"] = series.diff(2).shift(1)
    names += ["diff_1", "diff_2"]

    cal = calendar_features(series.index)
    for i, label in enumerate(
        ["tod_sin", "tod_cos", "dow_sin", "dow_cos", "is_weekend"]
    ):
        frame[label] = cal[:, i]
        names.append(label)

    matrix = frame[names].to_numpy(dtype="float64")
    matrix = np.nan_to_num(
        pd.DataFrame(matrix).bfill().ffill().to_numpy(dtype="float64"), nan=0.0
    )
    return matrix, names


# --------------------------------------------------------------------------
# Isolation Forest
# --------------------------------------------------------------------------


class IsolationForestDetector(BaseDetector):
    name = "isolation_forest"
    display_name = "Isolation Forest"
    family = "machine learning"
    forecasts = False
    description = (
        "Unsupervised ensemble of random isolation trees over a causal feature "
        "window (lags, rolling moments, differences and calendar context). "
        "Anomaly score is the negated path-depth decision function."
    )
    rationale = (
        "Included as the representative pure machine-learning detector. It "
        "assumes neither normality nor a generative model, so it can catch "
        "multivariate-shaped oddities - an ordinary level reached at an "
        "extraordinary rate, say - that a univariate residual threshold will "
        "miss. The trade-off is that it yields no forecast and therefore no "
        "capacity-planning value, and its scores are hard to explain to an "
        "on-call engineer."
    )

    def fit(self, train: pd.Series) -> "IsolationForestDetector":
        self._train = train
        self._train_len = len(train)
        contamination = float(self.options.get("contamination", 0.03))

        features, names = build_window_features(train)
        self._scaler = StandardScaler().fit(features)

        self._model = IsolationForest(
            n_estimators=300,
            max_samples="auto",
            contamination=contamination,
            random_state=RANDOM_SEED,
            n_jobs=1,
        ).fit(self._scaler.transform(features))

        train_scores = -self._model.decision_function(self._scaler.transform(features))
        self._train_scores = train_scores

        self.params = {
            "n_estimators": 300,
            "contamination": contamination,
            "n_features": len(names),
            "feature_names": names,
        }
        self.diagnostics = {
            "train_score_mean": round(float(train_scores.mean()), 4),
            "train_score_p99": round(float(np.percentile(train_scores, 99)), 4),
        }
        return self

    def score(self, test: pd.Series, predictions: Optional[np.ndarray]) -> np.ndarray:
        combined = pd.concat([self._train, test])
        features, _ = build_window_features(combined)
        features = features[self._train_len :]
        return -self._model.decision_function(self._scaler.transform(features))

    def threshold_for(self, scores: np.ndarray) -> float:
        """Cut-point taken as the stricter of two contamination quantiles.

        The training quantile alone is brittle. On a series whose training
        window is unusually regular, the forest's train scores occupy a razor
        thin band, and any ordinary distribution shift in the hold-out pushes
        the entire window past the cut-point -- the detector then flags
        everything, which is worse than useless. The scored-window quantile
        alone has the opposite flaw: it guarantees exactly ``contamination`` of
        points are flagged, so the model can never report all-clear.

        Taking the maximum keeps the training calibration whenever it is the
        stricter of the two, while capping the alert budget at the
        contamination prior the user set. ``contamination`` is a stated prior
        on the alert rate, not information about where the anomalies are, so
        this is not label leakage -- but it is a real limitation of the method
        and it is reported as one.
        """
        contamination = float(self.options.get("contamination", 0.03))
        q = 100.0 * (1.0 - contamination)
        train_cut = float(np.percentile(self._train_scores, q))
        window_cut = float(np.percentile(np.asarray(scores, dtype="float64"), q))
        return max(train_cut, window_cut)


# --------------------------------------------------------------------------
# Residual learners for the hybrid
# --------------------------------------------------------------------------


class _TorchLSTM:  # pragma: no cover - only runs where torch is installed
    """Small single-layer LSTM regressor over a residual lag window."""

    def __init__(self, n_features: int, hidden: int = 32, epochs: int = 120):
        self.n_features = n_features
        self.hidden = hidden
        self.epochs = epochs
        torch.manual_seed(RANDOM_SEED)

        class Net(nn.Module):
            def __init__(self, n_in: int, hidden: int):
                super().__init__()
                self.lstm = nn.LSTM(1, hidden, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                out, _ = self.lstm(x.unsqueeze(-1))
                return self.head(out[:, -1, :]).squeeze(-1)

        self.net = Net(n_features, hidden)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_TorchLSTM":
        xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        opt = torch.optim.Adam(self.net.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        self.net.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = loss_fn(self.net(xt), yt)
            loss.backward()
            opt.step()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(X, dtype=torch.float32)).numpy()


class ResidualLearner:
    """Nonlinear model of what the linear stage left behind.

    Parameters
    ----------
    kind : {"gbr", "mlp", "lstm"}
        ``gbr``  gradient-boosted regression trees (default; robust, fast,
                 no feature scaling needed, handles the heavy-tailed residual
                 distribution well).
        ``mlp``  feed-forward neural network over the same lag window.
        ``lstm`` recurrent network; requires PyTorch and silently degrades to
                 ``mlp`` when it is not installed.
    """

    AVAILABLE = ("gbr", "mlp", "lstm")

    def __init__(self, kind: str = "gbr", lags: int = 12, random_state: int = RANDOM_SEED):
        if kind not in self.AVAILABLE:
            raise ValueError(f"Unknown residual learner '{kind}'.")
        self.requested_kind = kind
        self.kind = kind
        self.lags = int(lags)
        self.random_state = random_state
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.fallback_reason: Optional[str] = None

        if kind == "lstm" and not TORCH_AVAILABLE:
            self.kind = "mlp"
            self.fallback_reason = (
                "PyTorch is not installed, so the LSTM residual stage fell back "
                "to the feed-forward network. Install torch to enable it."
            )

    # -- design matrix -----------------------------------------------------

    def _design(
        self, residuals: np.ndarray, index: pd.DatetimeIndex
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build (X, y, valid_mask) from a residual sequence.

        Row *t* holds residuals t-1 .. t-lags plus calendar features for *t*.
        The target is the residual at *t*. Nothing at or after *t* enters X.
        """
        n = len(residuals)
        rows: List[np.ndarray] = []
        targets: List[float] = []
        valid = np.zeros(n, dtype=bool)

        cal = calendar_features(index)
        for t in range(self.lags, n):
            window = residuals[t - self.lags : t][::-1]
            rows.append(np.concatenate([window, cal[t]]))
            targets.append(residuals[t])
            valid[t] = True

        if not rows:
            return np.empty((0, self.lags + 5)), np.empty(0), valid
        return np.asarray(rows), np.asarray(targets), valid

    def _lag_only(self, X: np.ndarray) -> np.ndarray:
        return X[:, : self.lags]

    # -- API ---------------------------------------------------------------

    def fit(self, residuals: np.ndarray, index: pd.DatetimeIndex) -> "ResidualLearner":
        X, y, _ = self._design(np.asarray(residuals, dtype="float64"), index)
        if X.shape[0] < 30:
            self.model = None
            self.fallback_reason = (
                "Too few training residuals for a nonlinear stage; the hybrid "
                "reduces to its linear component."
            )
            return self

        if self.kind == "gbr":
            self.model = GradientBoostingRegressor(
                n_estimators=250,
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=self.random_state,
            ).fit(X, y)
        elif self.kind == "mlp":
            self.scaler = StandardScaler().fit(X)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = MLPRegressor(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    learning_rate_init=0.005,
                    max_iter=600,
                    early_stopping=True,
                    n_iter_no_change=25,
                    random_state=self.random_state,
                ).fit(self.scaler.transform(X), y)
        else:  # lstm
            self.scaler = StandardScaler().fit(self._lag_only(X))
            self.model = _TorchLSTM(self.lags).fit(
                self.scaler.transform(self._lag_only(X)), y
            )

        return self

    def predict(self, residuals: np.ndarray, index: pd.DatetimeIndex) -> np.ndarray:
        """Predict the residual at each point from its own recent history."""
        residuals = np.asarray(residuals, dtype="float64")
        out = np.zeros(len(residuals), dtype="float64")
        if self.model is None:
            return out

        X, _, valid = self._design(residuals, index)
        if X.shape[0] == 0:
            return out

        if self.kind == "gbr":
            pred = self.model.predict(X)
        elif self.kind == "mlp":
            pred = self.model.predict(self.scaler.transform(X))
        else:
            pred = self.model.predict(self.scaler.transform(self._lag_only(X)))

        out[valid] = pred
        return out

    def describe(self) -> Dict[str, object]:
        label = {
            "gbr": "Gradient-boosted regression trees",
            "mlp": "Feed-forward neural network (64-32, ReLU)",
            "lstm": "LSTM recurrent network (32 hidden units)",
        }[self.kind]
        return {
            "requested": self.requested_kind,
            "used": self.kind,
            "label": label,
            "lags": self.lags,
            "torch_available": TORCH_AVAILABLE,
            "fallback_reason": self.fallback_reason,
            "active": self.model is not None,
        }
