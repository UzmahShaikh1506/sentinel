"""
SENTINEL-TS :: Exploratory data analysis.

Produces every number and array the console's EDA tab renders:

* STL decomposition into trend / seasonal / remainder, plus the share of
  variance each component explains.
* Stationarity evidence from two tests with opposite null hypotheses (ADF and
  KPSS). Agreement between them is what licenses a differencing decision; the
  two disagreeing is itself a finding worth reporting.
* ACF and PACF with Bartlett confidence bands, which is how the ARMA orders
  below are justified rather than guessed.
* A daily seasonal profile and a weekday/hour heatmap.
* Distribution and volatility summaries.

Everything is returned as plain Python types so the API can serialise it
directly to JSON.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from .preprocessing import seasonal_profile


def _finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return arr[np.isfinite(arr)]


def _round_list(values, digits: int = 6) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in np.asarray(values, dtype="float64"):
        out.append(None if not np.isfinite(v) else round(float(v), digits))
    return out


# --------------------------------------------------------------------------
# Decomposition
# --------------------------------------------------------------------------


def decompose(series: pd.Series, period: int) -> Dict[str, object]:
    """STL decomposition with variance attribution.

    STL is preferred over the classical additive decomposition because it is
    robust to the outliers this project is explicitly looking for: a spike
    distorts a centred moving average far more than it distorts a locally
    weighted regression fit.
    """
    n = len(series)
    if period < 2 or n < 2 * period + 1:
        return {
            "available": False,
            "reason": (
                f"Series of length {n} is too short for a seasonal period of "
                f"{period}; STL needs at least {2 * period + 1} observations."
            ),
        }

    seasonal_smoother = max(7, (period // 2) * 2 + 1)
    stl = STL(series, period=period, seasonal=seasonal_smoother, robust=True)
    result = stl.fit()

    trend = result.trend
    seasonal = result.seasonal
    resid = result.resid

    total_var = float(np.nanvar(series.to_numpy()))
    if total_var <= 0:
        shares = {"trend": 0.0, "seasonal": 0.0, "remainder": 0.0}
    else:
        shares = {
            "trend": float(np.nanvar(trend.to_numpy()) / total_var),
            "seasonal": float(np.nanvar(seasonal.to_numpy()) / total_var),
            "remainder": float(np.nanvar(resid.to_numpy()) / total_var),
        }

    # Hyndman-Athanasopoulos strength measures, bounded to [0, 1].
    var_rs = float(np.nanvar((resid + seasonal).to_numpy()))
    var_rt = float(np.nanvar((resid + trend).to_numpy()))
    var_r = float(np.nanvar(resid.to_numpy()))
    seasonal_strength = max(0.0, 1.0 - var_r / var_rs) if var_rs > 0 else 0.0
    trend_strength = max(0.0, 1.0 - var_r / var_rt) if var_rt > 0 else 0.0

    return {
        "available": True,
        "period": int(period),
        "timestamps": [ts.isoformat() for ts in series.index],
        "observed": _round_list(series.to_numpy()),
        "trend": _round_list(trend.to_numpy()),
        "seasonal": _round_list(seasonal.to_numpy()),
        "remainder": _round_list(resid.to_numpy()),
        "variance_share": {k: round(v, 4) for k, v in shares.items()},
        "seasonal_strength": round(seasonal_strength, 4),
        "trend_strength": round(trend_strength, 4),
        "interpretation": _describe_decomposition(
            trend_strength, seasonal_strength, shares
        ),
    }


def _describe_decomposition(
    trend_strength: float, seasonal_strength: float, shares: Dict[str, float]
) -> str:
    bits: List[str] = []
    if seasonal_strength >= 0.6:
        bits.append(
            "A strong daily cycle dominates the signal, so any detector must "
            "model seasonality rather than threshold the raw level."
        )
    elif seasonal_strength >= 0.3:
        bits.append(
            "A moderate daily cycle is present; seasonal adjustment improves "
            "detection but does not drive the series on its own."
        )
    else:
        bits.append(
            "Seasonality is weak, so the series behaves closer to a noisy level "
            "process than to a repeating daily profile."
        )

    if trend_strength >= 0.6:
        bits.append(
            "The trend component is strong, indicating slow drift in the "
            "operating level over the observation window."
        )
    elif trend_strength < 0.25:
        bits.append("The level is broadly stable across the window.")

    if shares.get("remainder", 0) > 0.4:
        bits.append(
            "A large remainder share means much of the variation is irregular - "
            "precisely the component in which incidents live."
        )
    return " ".join(bits)


# --------------------------------------------------------------------------
# Stationarity
# --------------------------------------------------------------------------


def stationarity_tests(series: pd.Series) -> Dict[str, object]:
    """ADF and KPSS, plus a reconciled verdict.

    ADF  : H0 = a unit root is present (non-stationary).
    KPSS : H0 = the series is stationary around a deterministic level.

    Reading them together avoids the classic error of declaring stationarity
    from a single test.
    """
    values = _finite(series.to_numpy())
    out: Dict[str, object] = {}

    try:
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(
            values, autolag="AIC"
        )
        out["adf"] = {
            "statistic": round(float(adf_stat), 4),
            "p_value": round(float(adf_p), 6),
            "lags_used": int(adf_lags),
            "n_obs": int(adf_nobs),
            "critical_values": {k: round(float(v), 4) for k, v in adf_crit.items()},
            "null_hypothesis": "A unit root is present (series is non-stationary).",
            "reject_null": bool(adf_p < 0.05),
        }
    except Exception as exc:  # pragma: no cover - numerical guard
        out["adf"] = {"error": str(exc)}

    try:
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(
                values, regression="c", nlags="auto"
            )
        out["kpss"] = {
            "statistic": round(float(kpss_stat), 4),
            "p_value": round(float(kpss_p), 6),
            "lags_used": int(kpss_lags),
            "critical_values": {k: round(float(v), 4) for k, v in kpss_crit.items()},
            "null_hypothesis": "The series is stationary around a constant level.",
            "reject_null": bool(kpss_p < 0.05),
            "p_value_note": (
                "KPSS p-values are interpolated from a lookup table and are "
                "reported at the boundary when the statistic falls outside it."
            ),
        }
    except Exception as exc:  # pragma: no cover
        out["kpss"] = {"error": str(exc)}

    out["verdict"] = _reconcile_stationarity(out)
    return out


def _reconcile_stationarity(tests: Dict[str, object]) -> Dict[str, object]:
    adf = tests.get("adf", {})
    kpss_res = tests.get("kpss", {})
    if "error" in adf or "error" in kpss_res:
        return {
            "label": "inconclusive",
            "recommended_d": 1,
            "explanation": "At least one test failed to converge.",
        }

    adf_stationary = bool(adf.get("reject_null"))
    kpss_stationary = not bool(kpss_res.get("reject_null"))

    if adf_stationary and kpss_stationary:
        return {
            "label": "stationary",
            "recommended_d": 0,
            "explanation": (
                "ADF rejects the unit root and KPSS fails to reject stationarity. "
                "Both tests agree, so the series can be modelled without "
                "differencing (d = 0)."
            ),
        }
    if not adf_stationary and not kpss_stationary:
        return {
            "label": "non-stationary",
            "recommended_d": 1,
            "explanation": (
                "ADF cannot reject a unit root and KPSS rejects stationarity. "
                "Both tests agree the level wanders, so first differencing "
                "(d = 1) is indicated."
            ),
        }
    if adf_stationary and not kpss_stationary:
        return {
            "label": "trend-stationary",
            "recommended_d": 0,
            "explanation": (
                "ADF rejects the unit root while KPSS rejects level "
                "stationarity. This pattern points to stationarity around a "
                "deterministic trend; de-trending is preferable to differencing."
            ),
        }
    return {
        "label": "difference-stationary",
        "recommended_d": 1,
        "explanation": (
            "ADF cannot reject a unit root while KPSS fails to reject "
            "stationarity. The evidence is weak in both directions, typical of "
            "a near-unit-root process; d = 1 is the conservative choice."
        ),
    }


# --------------------------------------------------------------------------
# Correlation structure
# --------------------------------------------------------------------------


def correlation_structure(series: pd.Series, nlags: int = 60) -> Dict[str, object]:
    """ACF and PACF with Bartlett bands and a suggested ARMA order."""
    values = _finite(series.to_numpy())
    nlags = int(min(nlags, max(5, len(values) // 3)))

    acf_vals = acf(values, nlags=nlags, fft=True)
    try:
        pacf_vals = pacf(values, nlags=min(nlags, len(values) // 2 - 1), method="ywm")
    except Exception:  # pragma: no cover
        pacf_vals = np.full(nlags + 1, np.nan)

    band = 1.96 / np.sqrt(len(values))

    # First lag at which each function falls inside the band -> candidate order.
    def _cutoff(arr: np.ndarray) -> int:
        for lag in range(1, len(arr)):
            if np.isfinite(arr[lag]) and abs(arr[lag]) < band:
                return lag - 1
            if lag >= 12:
                break
        return min(3, len(arr) - 1)

    suggested_p = int(max(0, min(3, _cutoff(pacf_vals))))
    suggested_q = int(max(0, min(3, _cutoff(acf_vals))))

    return {
        "nlags": nlags,
        "acf": _round_list(acf_vals),
        "pacf": _round_list(pacf_vals),
        "confidence_band": round(float(band), 6),
        "suggested_p": suggested_p,
        "suggested_q": suggested_q,
        "interpretation": (
            f"The PACF falls inside the +/-{band:.3f} band after lag "
            f"{suggested_p} and the ACF after lag {suggested_q}, which is the "
            f"evidence for the candidate orders p = {suggested_p}, "
            f"q = {suggested_q} carried into model selection. A slowly decaying "
            "ACF would instead indicate differencing is required first."
        ),
    }


# --------------------------------------------------------------------------
# Seasonality views
# --------------------------------------------------------------------------


def seasonality_views(series: pd.Series, period: int) -> Dict[str, object]:
    """Daily profile plus an hour-of-day by day-of-week heatmap."""
    profile = seasonal_profile(series, period)

    frame = pd.DataFrame(
        {
            "value": series.to_numpy(),
            "hour": series.index.hour,
            "dow": series.index.dayofweek,
        }
    )
    pivot = frame.pivot_table(
        index="dow", columns="hour", values="value", aggfunc="mean"
    )
    pivot = pivot.reindex(index=range(7), columns=range(24))

    heatmap: List[List[Optional[float]]] = []
    for dow in range(7):
        row: List[Optional[float]] = []
        for hour in range(24):
            val = pivot.loc[dow, hour] if dow in pivot.index else np.nan
            row.append(None if pd.isna(val) else round(float(val), 4))
        heatmap.append(row)

    hourly = frame.groupby("hour")["value"].mean()
    peak_hour = int(hourly.idxmax()) if not hourly.empty else None
    trough_hour = int(hourly.idxmin()) if not hourly.empty else None

    return {
        "daily_profile": _round_list(profile.to_numpy()) if not profile.empty else [],
        "profile_period": int(period),
        "heatmap": heatmap,
        "heatmap_rows": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        "peak_hour": peak_hour,
        "trough_hour": trough_hour,
        "interpretation": (
            f"Load peaks around {peak_hour:02d}:00 and bottoms out around "
            f"{trough_hour:02d}:00 UTC. A fixed threshold calibrated at the peak "
            "would be blind to an incident at the trough, which is the core "
            "argument for seasonal modelling."
            if peak_hour is not None
            else "Insufficient coverage for an hourly profile."
        ),
    }


# --------------------------------------------------------------------------
# Distribution and volatility
# --------------------------------------------------------------------------


def distribution_summary(series: pd.Series) -> Dict[str, object]:
    values = _finite(series.to_numpy())
    if values.size == 0:
        return {}

    counts, edges = np.histogram(values, bins=40)
    diffs = np.diff(values)
    roll_std = pd.Series(values).rolling(48, min_periods=12).std()

    return {
        "count": int(values.size),
        "mean": round(float(np.mean(values)), 4),
        "median": round(float(np.median(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "q01": round(float(np.percentile(values, 1)), 4),
        "q25": round(float(np.percentile(values, 25)), 4),
        "q75": round(float(np.percentile(values, 75)), 4),
        "q99": round(float(np.percentile(values, 99)), 4),
        "skewness": round(float(pd.Series(values).skew()), 4),
        "kurtosis": round(float(pd.Series(values).kurt()), 4),
        "histogram": {
            "counts": [int(c) for c in counts],
            "edges": _round_list(edges, 4),
        },
        "volatility": {
            "diff_std": round(float(np.std(diffs)), 4) if diffs.size else 0.0,
            "rolling_std_mean": (
                round(float(roll_std.mean()), 4) if roll_std.notna().any() else 0.0
            ),
            "rolling_std_max": (
                round(float(roll_std.max()), 4) if roll_std.notna().any() else 0.0
            ),
            "heteroskedastic": bool(
                roll_std.notna().any()
                and roll_std.max() > 2.5 * max(roll_std.mean(), 1e-9)
            ),
        },
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


def run_eda(series: pd.Series, period: int, max_points: int = 1600) -> Dict[str, object]:
    """Full EDA bundle for one preprocessed series."""
    result: Dict[str, object] = {
        "n_obs": int(len(series)),
        "period": int(period),
    }

    result["decomposition"] = decompose(series, period)
    result["stationarity"] = stationarity_tests(series)
    result["correlation"] = correlation_structure(series)
    result["seasonality"] = seasonality_views(series, period)
    result["distribution"] = distribution_summary(series)

    # Series for plotting, thinned if very long so the payload stays small.
    step = max(1, len(series) // max_points)
    thinned = series.iloc[::step]
    result["series"] = {
        "timestamps": [ts.isoformat() for ts in thinned.index],
        "values": _round_list(thinned.to_numpy(), 4),
        "thinning_step": step,
    }

    insights = [
        result["decomposition"].get("interpretation", ""),
        result["stationarity"]["verdict"]["explanation"],
        result["correlation"]["interpretation"],
        result["seasonality"]["interpretation"],
    ]
    dist = result["distribution"]
    if dist.get("volatility", {}).get("heteroskedastic"):
        insights.append(
            "Rolling volatility is far from constant, so the variance itself "
            "changes over time. A detector with a fixed global threshold will "
            "over-alert in volatile stretches and under-alert in calm ones; "
            "this motivates the rolling robust scale used throughout."
        )
    if abs(dist.get("skewness", 0.0)) > 1.0:
        insights.append(
            f"The distribution is markedly skewed (skew = {dist['skewness']}), "
            "so symmetric Gaussian thresholds are not appropriate and robust "
            "statistics are used instead."
        )
    result["insights"] = [s for s in insights if s]

    return result
