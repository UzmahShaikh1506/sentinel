"""
SENTINEL-TS :: Preprocessing.

Turns a raw, irregular telemetry dump into a gap-free, regularly spaced series
that classical time-series machinery can accept, while recording an audit trail
of every change made. The audit trail matters: in an operations setting you
must be able to say which values the model saw and which were reconstructed.

Design notes
------------
* **Duplicate timestamps** are collapsed by mean, not dropped, because agent
  double-reporting is common and both readings are informative.
* **Resampling** uses the mean within each bucket. For counters (requests,
  bytes) a sum would also be defensible; the mean keeps the unit interpretable
  and the two differ only by a constant factor at fixed bucket width.
* **Gaps** are filled by time-aware interpolation for short runs and by the
  seasonal profile for long runs, so a multi-hour outage is not papered over
  with a straight line that would itself look like an anomaly.
* **Outliers are never removed.** In anomaly detection the extremes *are* the
  signal. They are flagged and carried forward; only the model's training
  statistics are made robust to them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import PERIODS_PER_DAY

#: Gaps at most this many buckets long are interpolated linearly in time.
SHORT_GAP_BUCKETS = 6


@dataclass
class PreprocessReport:
    """Audit trail for one preprocessing run."""

    n_input: int = 0
    n_output: int = 0
    n_duplicate_timestamps: int = 0
    n_non_finite: int = 0
    n_missing_after_resample: int = 0
    n_filled_interpolated: int = 0
    n_filled_seasonal: int = 0
    n_outliers_flagged: int = 0
    outlier_rate: float = 0.0
    freq: str = ""
    seasonal_period: int = 0
    start: str = ""
    end: str = ""
    duration_days: float = 0.0
    coverage_pct: float = 100.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "n_input": self.n_input,
            "n_output": self.n_output,
            "n_duplicate_timestamps": self.n_duplicate_timestamps,
            "n_non_finite": self.n_non_finite,
            "n_missing_after_resample": self.n_missing_after_resample,
            "n_filled_interpolated": self.n_filled_interpolated,
            "n_filled_seasonal": self.n_filled_seasonal,
            "n_outliers_flagged": self.n_outliers_flagged,
            "outlier_rate": round(self.outlier_rate, 4),
            "freq": self.freq,
            "seasonal_period": self.seasonal_period,
            "start": self.start,
            "end": self.end,
            "duration_days": round(self.duration_days, 2),
            "coverage_pct": round(self.coverage_pct, 2),
            "notes": self.notes,
        }


# --------------------------------------------------------------------------
# Robust statistics
# --------------------------------------------------------------------------

#: 1 / Phi^-1(0.75); scales the median absolute deviation to be a consistent
#: estimator of sigma for Gaussian data.
MAD_TO_SIGMA = 1.4826


def robust_scale(values: np.ndarray) -> float:
    """Median absolute deviation, rescaled to a standard-deviation equivalent.

    Falls back to the inter-quartile range and then to the plain standard
    deviation when the MAD collapses to zero, which happens on series with long
    constant runs (an idle host reporting exactly 0.0, for instance).
    """
    arr = np.asarray(values, dtype="float64")
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    mad = np.median(np.abs(arr - np.median(arr))) * MAD_TO_SIGMA
    if mad > 1e-12:
        return float(mad)
    iqr = float(np.subtract(*np.percentile(arr, [75, 25]))) / 1.349
    if iqr > 1e-12:
        return iqr
    std = float(np.std(arr))
    return std if std > 1e-12 else 1e-9


def robust_zscore(values: np.ndarray) -> np.ndarray:
    """Global robust z-score: (x - median) / MAD-sigma."""
    arr = np.asarray(values, dtype="float64")
    centre = float(np.nanmedian(arr))
    scale = robust_scale(arr)
    return (arr - centre) / scale


def rolling_robust_zscore(
    series: pd.Series, window: int, min_periods: int | None = None
) -> pd.Series:
    """Causal rolling robust z-score.

    The score at time *t* uses a trailing window and nothing else, so this is
    valid for online scoring as well as offline evaluation.

    Strict causality is harder than it looks. An earlier version of this
    function fell back to a *global* MAD when a window degenerated, and to a
    backward fill during warm-up. Both quietly let information from later in
    the series reach earlier scores -- the unit test
    ``test_rolling_zscore_is_causal`` catches exactly that. The fallbacks here
    are expanding rather than global, and the warm-up period is scored as zero
    (never alert) instead of being filled from the future.
    """
    if min_periods is None:
        min_periods = max(8, window // 4)

    roll = series.rolling(window=window, min_periods=min_periods)
    centre = roll.median()
    deviation = (series - centre).abs()
    mad = deviation.rolling(window=window, min_periods=min_periods).median() * MAD_TO_SIGMA

    # Expanding (past-only) fallback for windows whose MAD collapses to zero,
    # which happens on long constant runs such as an idle host reporting 0.0.
    expanding_centre = series.expanding(min_periods=min_periods).median()
    expanding_mad = (
        (series - expanding_centre)
        .abs()
        .expanding(min_periods=min_periods)
        .median()
        * MAD_TO_SIGMA
    )

    mad = mad.where(mad > 1e-9, expanding_mad)
    mad = mad.where(mad > 1e-9, 1e-9)

    z = (series - centre) / mad
    # Warm-up (before min_periods observations) is genuinely unscoreable.
    return z.replace([np.inf, -np.inf], np.nan).fillna(0.0)


# --------------------------------------------------------------------------
# Seasonal profile
# --------------------------------------------------------------------------


def seasonal_profile(series: pd.Series, period: int) -> pd.Series:
    """Median value for each position in the seasonal cycle.

    Used both for long-gap imputation and as the seasonal-naive baseline's view
    of a 'typical' day.
    """
    if period <= 1 or series.dropna().empty:
        return pd.Series(dtype="float64")
    pos = np.arange(len(series)) % period
    frame = pd.DataFrame({"pos": pos, "value": series.to_numpy()})
    return frame.groupby("pos")["value"].median()


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------


def preprocess(
    series: pd.Series,
    freq: str = "15min",
    outlier_sigma: float = 4.0,
) -> Tuple[pd.Series, pd.Series, PreprocessReport]:
    """Clean and regularise a raw series.

    Returns
    -------
    clean : pd.Series
        Gap-free series on a regular ``freq`` grid.
    flags : pd.Series
        Boolean mask, aligned to ``clean``, marking statistically extreme
        observations. These are *not* removed -- the flag exists so the UI can
        shade them during EDA and so model fitting can down-weight them.
    report : PreprocessReport
        Audit trail.
    """
    report = PreprocessReport(freq=freq)
    report.n_input = int(len(series))

    if series.empty:
        raise ValueError("Cannot preprocess an empty series.")
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series must be indexed by a DatetimeIndex.")

    work = series.sort_index().astype("float64")

    # ---- non-finite values -------------------------------------------------
    non_finite = ~np.isfinite(work.to_numpy())
    report.n_non_finite = int(non_finite.sum())
    if report.n_non_finite:
        work = work.mask(non_finite)
        report.notes.append(
            f"{report.n_non_finite} non-finite reading(s) masked before resampling."
        )

    # ---- duplicate timestamps ---------------------------------------------
    dup_count = int(work.index.duplicated().sum())
    report.n_duplicate_timestamps = dup_count
    if dup_count:
        work = work.groupby(level=0).mean()
        report.notes.append(
            f"{dup_count} duplicate timestamp(s) collapsed by mean."
        )

    # ---- resample to a regular grid ---------------------------------------
    resampled = work.resample(freq).mean()
    report.n_missing_after_resample = int(resampled.isna().sum())

    period = PERIODS_PER_DAY.get(freq, 96)
    report.seasonal_period = period

    # ---- gap filling -------------------------------------------------------
    filled, n_interp, n_seasonal = _fill_gaps(resampled, period)
    report.n_filled_interpolated = n_interp
    report.n_filled_seasonal = n_seasonal
    if n_interp:
        report.notes.append(
            f"{n_interp} bucket(s) in short gaps (<= {SHORT_GAP_BUCKETS}) "
            "filled by time interpolation."
        )
    if n_seasonal:
        report.notes.append(
            f"{n_seasonal} bucket(s) in long gaps reconstructed from the "
            "seasonal profile rather than interpolated."
        )

    # ---- outlier flagging (not removal) -----------------------------------
    z = rolling_robust_zscore(filled, window=min(period, max(12, len(filled) // 10)))
    flags = z.abs() > outlier_sigma
    report.n_outliers_flagged = int(flags.sum())
    report.outlier_rate = float(flags.mean()) if len(flags) else 0.0
    report.notes.append(
        f"{report.n_outliers_flagged} observation(s) exceed |robust z| > "
        f"{outlier_sigma}; retained as signal, flagged for review."
    )

    # ---- summary -----------------------------------------------------------
    report.n_output = int(len(filled))
    report.start = filled.index[0].isoformat()
    report.end = filled.index[-1].isoformat()
    span = filled.index[-1] - filled.index[0]
    report.duration_days = span.total_seconds() / 86400.0
    observed = report.n_output - report.n_missing_after_resample
    report.coverage_pct = (
        100.0 * observed / report.n_output if report.n_output else 100.0
    )

    filled.name = "value"
    flags.name = "outlier"
    return filled, flags, report


def _fill_gaps(series: pd.Series, period: int) -> Tuple[pd.Series, int, int]:
    """Fill NaN runs: short ones by interpolation, long ones seasonally."""
    if not series.isna().any():
        return series, 0, 0

    filled = series.copy()
    is_na = series.isna()

    # Identify contiguous NaN runs.
    run_id = (is_na != is_na.shift()).cumsum()
    runs = [
        (grp.index[0], grp.index[-1], len(grp))
        for _, grp in series.groupby(run_id)
        if grp.isna().all()
    ]

    profile = seasonal_profile(series, period)
    positions = pd.Series(np.arange(len(series)) % period, index=series.index)

    n_interp = 0
    n_seasonal = 0
    long_gap_slices = []

    for start, end, length in runs:
        if length <= SHORT_GAP_BUCKETS:
            n_interp += length
        else:
            n_seasonal += length
            long_gap_slices.append((start, end))

    # Seasonal reconstruction first, so interpolation does not bridge the gap.
    if long_gap_slices and not profile.empty:
        for start, end in long_gap_slices:
            idx = filled.loc[start:end].index
            filled.loc[idx] = positions.loc[idx].map(profile).to_numpy()

    # Remaining short gaps.
    filled = filled.interpolate(method="time", limit_direction="both")

    # Any residual NaN (for example a wholly empty profile) falls back to median.
    if filled.isna().any():
        filled = filled.fillna(float(series.median()))

    return filled, n_interp, n_seasonal


def train_test_split(
    series: pd.Series, test_size: float = 0.30
) -> Tuple[pd.Series, pd.Series]:
    """Chronological hold-out split.

    Random splitting would leak future information through the autocorrelation
    structure, so the split is strictly by time.
    """
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must lie in (0, 1).")
    n_test = max(8, int(round(len(series) * test_size)))
    if n_test >= len(series):
        raise ValueError("test_size leaves no training data.")
    split = len(series) - n_test
    return series.iloc[:split], series.iloc[split:]


def windows_to_point_labels(
    index: pd.DatetimeIndex,
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]],
) -> np.ndarray:
    """Expand ground-truth anomaly windows into a boolean point mask."""
    labels = np.zeros(len(index), dtype=bool)
    for start, end in windows:
        labels |= (index >= start) & (index <= end)
    return labels
