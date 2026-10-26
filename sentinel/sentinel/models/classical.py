"""
SENTINEL-TS :: Classical time-series models.

``HoltWinters``
    Additive error / damped additive trend / additive seasonal ETS. Captures
    level, drift and the daily cycle with a small, interpretable state vector.

``SarimaxFourier``
    ARMA dynamics on top of harmonic seasonal regressors, with the order chosen
    by an AIC grid search rather than asserted. At 15-minute buckets the daily
    period is m = 96; a textbook SARIMA would need 96 seasonal lags, which is
    not estimable here, so the seasonality moves into Fourier exogenous terms.

Both share one important implementation detail. Parameters are estimated on the
training window only, and the hold-out predictions are produced by *re-running
the filter* over the full series with those frozen parameters. This yields
genuine one-step-ahead forecasts on the test set without ever re-estimating on
data the model is about to be judged on.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .base import BaseDetector, fourier_terms


class HoltWinters(BaseDetector):
    name = "holt_winters"
    display_name = "Holt-Winters ETS"
    family = "classical"
    forecasts = True
    description = (
        "Exponential smoothing state-space model with additive error, damped "
        "additive trend and additive daily seasonality; alerts on the robust "
        "z-score of the one-step-ahead residual."
    )
    rationale = (
        "Chosen because the EDA shows a strong, stable daily profile with slow "
        "level drift, which is exactly the structure ETS encodes. Damping the "
        "trend prevents the multi-step forecast from extrapolating a short "
        "local slope into an implausible ramp."
    )

    def fit(self, train: pd.Series) -> "HoltWinters":
        self._train = train
        self._train_len = len(train)

        seasonal_periods = self.period if len(train) >= 2 * self.period + 10 else None
        self._seasonal_periods = seasonal_periods
        self._spec = dict(
            error="add",
            trend="add",
            damped_trend=True,
            seasonal="add" if seasonal_periods else None,
            seasonal_periods=seasonal_periods,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._result = ETSModel(train, **self._spec).fit(disp=False)

        resid = train.to_numpy(dtype="float64") - self._result.fittedvalues.to_numpy(
            dtype="float64"
        )
        self._calibrate_residuals(resid[self.period :] if len(resid) > self.period else resid)

        params = self._result.params
        self.params = {
            "error": "additive",
            "trend": "additive (damped)",
            "seasonal": "additive" if seasonal_periods else "none",
            "seasonal_periods": seasonal_periods,
            "smoothing_level_alpha": round(float(params[0]), 4),
            "smoothing_trend_beta": round(float(params[1]), 4) if len(params) > 1 else None,
            "aic": round(float(self._result.aic), 2),
            "bic": round(float(self._result.bic), 2),
        }
        self.diagnostics = {
            "log_likelihood": round(float(self._result.llf), 2),
            "n_train": int(len(train)),
        }
        return self

    def predict(self, test: pd.Series) -> Optional[np.ndarray]:
        """One-step-ahead forecasts with train-estimated parameters frozen.

        ``smooth`` re-runs the state recursion over the combined series using
        the supplied parameter vector. Because the parameters came from the
        training fit, nothing in the test window informs them.
        """
        combined = pd.concat([self._train, test])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            smoothed = ETSModel(combined, **self._spec).smooth(self._result.params)
        return smoothed.fittedvalues.iloc[self._train_len :].to_numpy(dtype="float64")

    def forecast_future(self, horizon: int) -> Optional[Dict[str, object]]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                values = np.asarray(self._result.forecast(horizon), dtype="float64")
            return {
                "values": [round(float(v), 4) for v in values],
                "method": "ETS state-space multi-step forecast",
            }
        except Exception:  # pragma: no cover
            return None


class SarimaxFourier(BaseDetector):
    name = "sarimax_fourier"
    display_name = "SARIMAX + Fourier"
    family = "classical"
    forecasts = True
    description = (
        "ARMA(p, d, q) dynamics estimated jointly with harmonic regressors for "
        "the daily cycle. Order selected by AIC over a candidate grid informed "
        "by the ACF/PACF cut-offs."
    )
    rationale = (
        "A direct seasonal ARIMA is impractical here: at 15-minute resolution "
        "the daily period is 96, so SARIMA(p,d,q)(P,D,Q)[96] would require 96 "
        "seasonal lags from roughly 900 training observations. Encoding the "
        "cycle as four Fourier harmonic pairs captures the same periodic shape "
        "in eight coefficients and leaves the ARMA part to model only the "
        "short-memory dynamics it is good at."
    )

    #: Candidate orders. Kept small and defensible rather than exhaustive: a
    #: wide grid on a short series invites overfitting the selection criterion.
    CANDIDATE_ORDERS: List[Tuple[int, int, int]] = [
        (1, 0, 1),
        (2, 0, 1),
        (2, 0, 2),
        (3, 0, 1),
        (1, 1, 1),
        (2, 1, 2),
    ]

    def fit(self, train: pd.Series) -> "SarimaxFourier":
        self._train = train
        self._train_len = len(train)
        self._origin = train.index[0]
        self._k = int(self.options.get("fourier_k", 4))

        exog_train = fourier_terms(train.index, 24.0, self._k, origin=self._origin)

        suggested = self.options.get("suggested_order")
        candidates = list(self.CANDIDATE_ORDERS)
        if suggested and tuple(suggested) not in candidates:
            candidates.insert(0, tuple(suggested))

        search: List[Dict[str, object]] = []
        best = None
        for order in candidates:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = SARIMAX(
                        train,
                        exog=exog_train,
                        order=order,
                        trend="c" if order[1] == 0 else "n",
                        enforce_stationarity=True,
                        enforce_invertibility=True,
                    ).fit(disp=False, maxiter=120, method="lbfgs")
                aic = float(res.aic)
                if not np.isfinite(aic):
                    continue
                search.append(
                    {
                        "order": list(order),
                        "aic": round(aic, 2),
                        "bic": round(float(res.bic), 2),
                    }
                )
                if best is None or aic < best[0]:
                    best = (aic, order, res)
            except Exception:  # pragma: no cover - skip unstable candidates
                search.append({"order": list(order), "aic": None, "bic": None})

        if best is None:
            raise RuntimeError("No SARIMAX candidate converged.")

        self._aic, self._order, self._result = best

        resid = np.asarray(self._result.resid, dtype="float64")
        burn = max(self._order[0], self._order[2], 5)
        self._calibrate_residuals(resid[burn:])

        self.params = {
            "order": list(self._order),
            "trend": "c" if self._order[1] == 0 else "n",
            "fourier_harmonics": self._k,
            "fourier_terms": 2 * self._k,
            "seasonal_period_hours": 24,
            "aic": round(self._aic, 2),
            "bic": round(float(self._result.bic), 2),
        }
        self.diagnostics = {
            "order_search": sorted(
                [s for s in search if s["aic"] is not None],
                key=lambda r: r["aic"],
            ),
            "selection_criterion": "AIC",
            "n_train": int(len(train)),
            "log_likelihood": round(float(self._result.llf), 2),
        }

        try:
            lb = self._result.test_serial_correlation(method="ljungbox", lags=[24])
            self.diagnostics["ljung_box_lag24_pvalue"] = round(float(lb[0, 1, 0]), 4)
            self.diagnostics["residual_autocorrelation_note"] = (
                "A Ljung-Box p-value above 0.05 means no residual "
                "autocorrelation remains at lag 24, i.e. the linear structure "
                "has been adequately extracted."
            )
        except Exception:  # pragma: no cover
            pass

        return self

    def predict(self, test: pd.Series) -> Optional[np.ndarray]:
        """One-step-ahead forecasts over the test window.

        ``append(..., refit=False)`` extends the Kalman filter through the new
        observations while holding every estimated parameter fixed, so the
        fitted values over the appended span are true out-of-sample one-step
        predictions.
        """
        exog_test = fourier_terms(test.index, 24.0, self._k, origin=self._origin)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            extended = self._result.append(test, exog=exog_test, refit=False)
        return extended.fittedvalues.iloc[-len(test) :].to_numpy(dtype="float64")

    def in_sample_residuals(self) -> np.ndarray:
        return np.asarray(self._result.resid, dtype="float64")

    def forecast_future(self, horizon: int) -> Optional[Dict[str, object]]:
        try:
            last = self._train.index[-1]
            step = self._train.index[1] - self._train.index[0]
            future_index = pd.date_range(last + step, periods=horizon, freq=step)
            exog_future = fourier_terms(future_index, 24.0, self._k, origin=self._origin)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fc = self._result.get_forecast(steps=horizon, exog=exog_future)
            mean = np.asarray(fc.predicted_mean, dtype="float64")
            ci = np.asarray(fc.conf_int(alpha=0.05), dtype="float64")
            return {
                "values": [round(float(v), 4) for v in mean],
                "lower": [round(float(v), 4) for v in ci[:, 0]],
                "upper": [round(float(v), 4) for v in ci[:, 1]],
                "method": "SARIMAX multi-step forecast with 95% prediction interval",
            }
        except Exception:  # pragma: no cover
            return None
