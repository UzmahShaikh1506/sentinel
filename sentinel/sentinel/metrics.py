"""
SENTINEL-TS :: Evaluation metrics.

Two families, because the project has two jobs.

Forecasting
    MAE, RMSE, MAPE, sMAPE, MASE and R^2. MASE is included because MAPE is
    undefined at zero and explodes near it -- a real hazard on request-count
    and byte-count telemetry that legitimately idles at zero overnight.

Detection
    Point-wise precision / recall / F1, plus three measures that matter far
    more operationally:

    * **Point-adjusted F1** -- the standard convention in the time-series
      anomaly literature. If any point inside a ground-truth window is flagged,
      the whole window counts as detected. This reflects how an on-call rota
      actually works: one page is enough.
    * **Detection latency** -- how long after an incident begins the first
      alert fires. A detector that is perfect but slow is still a bad pager.
    * **False alarms per day** -- the quantity that determines whether a team
      keeps the alert switched on.

Reporting only point-wise F1 would flatter detectors that fire continuously,
and reporting only point-adjusted F1 would hide alert fatigue; both are shown.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EPS = 1e-9


# --------------------------------------------------------------------------
# Forecast accuracy
# --------------------------------------------------------------------------


def forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    insample: Optional[np.ndarray] = None,
    seasonal_period: int = 1,
) -> Dict[str, Optional[float]]:
    """Point-forecast accuracy on the hold-out window."""
    a = np.asarray(actual, dtype="float64")
    p = np.asarray(predicted, dtype="float64")

    mask = np.isfinite(a) & np.isfinite(p)
    a, p = a[mask], p[mask]
    if a.size == 0:
        return {k: None for k in ("mae", "rmse", "mape", "smape", "mase", "r2", "n")}

    err = a - p
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # MAPE over non-zero actuals only; reported as None if nothing qualifies.
    nz = np.abs(a) > EPS
    mape = float(np.mean(abs_err[nz] / np.abs(a[nz])) * 100.0) if nz.any() else None

    denom = np.abs(a) + np.abs(p)
    smape_mask = denom > EPS
    smape = (
        float(np.mean(2.0 * abs_err[smape_mask] / denom[smape_mask]) * 100.0)
        if smape_mask.any()
        else None
    )

    # MASE scaled by the in-sample seasonal-naive error, per Hyndman & Koehler.
    mase: Optional[float] = None
    if insample is not None:
        ins = np.asarray(insample, dtype="float64")
        ins = ins[np.isfinite(ins)]
        m = max(1, int(seasonal_period))
        if ins.size > m:
            scale = float(np.mean(np.abs(ins[m:] - ins[:-m])))
            if scale > EPS:
                mase = float(mae / scale)

    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((a - np.mean(a)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > EPS else None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "smape": round(smape, 4) if smape is not None else None,
        "mase": round(mase, 4) if mase is not None else None,
        "r2": round(r2, 4) if r2 is not None else None,
        "n": int(a.size),
    }


# --------------------------------------------------------------------------
# Detection accuracy
# --------------------------------------------------------------------------


def _confusion(pred: np.ndarray, truth: np.ndarray) -> Tuple[int, int, int, int]:
    tp = int(np.sum(pred & truth))
    fp = int(np.sum(pred & ~truth))
    fn = int(np.sum(~pred & truth))
    tn = int(np.sum(~pred & ~truth))
    return tp, fp, fn, tn


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def point_adjust(pred: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Expand predictions to full ground-truth segments that were hit.

    The standard protocol from the deep anomaly-detection literature: a segment
    detected anywhere is credited in full. Applied only to true segments, so it
    never manufactures true positives outside them.
    """
    adjusted = pred.copy()
    in_segment = False
    start = 0
    for i, flag in enumerate(truth):
        if flag and not in_segment:
            in_segment, start = True, i
        elif not flag and in_segment:
            if pred[start:i].any():
                adjusted[start:i] = True
            in_segment = False
    if in_segment and pred[start:].any():
        adjusted[start:] = True
    return adjusted


def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Contiguous True runs as inclusive (start, end) index pairs."""
    out: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def detection_metrics(
    predicted: np.ndarray,
    truth: Optional[np.ndarray],
    index: Optional[pd.DatetimeIndex] = None,
) -> Dict[str, object]:
    """Full detection scorecard.

    When ``truth`` is unavailable (user upload with no labels) the supervised
    fields are returned as ``None`` and only the unsupervised alert-behaviour
    statistics are populated, so the UI never shows a fabricated score.
    """
    pred = np.asarray(predicted, dtype=bool)
    n = pred.size

    duration_days = 1.0
    if index is not None and len(index) > 1:
        duration_days = max(
            (index[-1] - index[0]).total_seconds() / 86400.0, 1e-6
        )

    alert_segments = _segments(pred)
    base: Dict[str, object] = {
        "n_points": int(n),
        "n_flagged": int(pred.sum()),
        "flag_rate": round(float(pred.mean()), 5) if n else 0.0,
        "n_alert_episodes": len(alert_segments),
        "alerts_per_day": round(len(alert_segments) / duration_days, 3),
        "labelled": truth is not None and bool(np.asarray(truth).any()),
    }

    if truth is None or not np.asarray(truth).any():
        base.update(
            {
                "precision": None,
                "recall": None,
                "f1": None,
                "pa_precision": None,
                "pa_recall": None,
                "pa_f1": None,
                "tp": None,
                "fp": None,
                "fn": None,
                "tn": None,
                "n_true_segments": 0,
                "n_segments_detected": None,
                "segment_recall": None,
                "mean_detection_latency_points": None,
                "mean_detection_latency_minutes": None,
                "false_alarms_per_day": base["alerts_per_day"],
            }
        )
        return base

    tru = np.asarray(truth, dtype=bool)
    tp, fp, fn, tn = _confusion(pred, tru)
    precision, recall, f1 = _prf(tp, fp, fn)

    adjusted = point_adjust(pred, tru)
    atp, afp, afn, _ = _confusion(adjusted, tru)
    pa_precision, pa_recall, pa_f1 = _prf(atp, afp, afn)

    true_segments = _segments(tru)
    detected = 0
    latencies: List[int] = []
    for start, end in true_segments:
        hits = np.where(pred[start : end + 1])[0]
        if hits.size:
            detected += 1
            latencies.append(int(hits[0]))

    bucket_minutes = 0.0
    if index is not None and len(index) > 1:
        bucket_minutes = float(
            (index[1] - index[0]).total_seconds() / 60.0
        )

    mean_latency = float(np.mean(latencies)) if latencies else None

    # False alarms = alert episodes with no overlap with any true segment.
    false_episodes = 0
    for start, end in alert_segments:
        if not tru[start : end + 1].any():
            false_episodes += 1

    base.update(
        {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "pa_precision": round(pa_precision, 4),
            "pa_recall": round(pa_recall, 4),
            "pa_f1": round(pa_f1, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "n_true_segments": len(true_segments),
            "n_segments_detected": detected,
            "segment_recall": (
                round(detected / len(true_segments), 4) if true_segments else None
            ),
            "mean_detection_latency_points": (
                round(mean_latency, 2) if mean_latency is not None else None
            ),
            "mean_detection_latency_minutes": (
                round(mean_latency * bucket_minutes, 1)
                if mean_latency is not None and bucket_minutes
                else None
            ),
            "false_alarms_per_day": round(false_episodes / duration_days, 3),
        }
    )
    return base


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------

#: Weights used to turn the scorecard into a single ordering for the
#: leaderboard. Detection quality dominates because that is the product's job;
#: forecast accuracy is a means, not the end. Alert burden carries a real
#: negative weight so a detector cannot win by flagging everything.
RANK_WEIGHTS = {
    "pa_f1": 0.45,
    "segment_recall": 0.25,
    "f1": 0.15,
    "latency": 0.10,
    "alert_burden": 0.05,
}


def composite_score(
    detection: Dict[str, object], forecast: Optional[Dict[str, object]] = None
) -> Optional[float]:
    """Single 0-1 operational score; ``None`` when the series is unlabelled."""
    if not detection.get("labelled"):
        return None

    pa_f1 = float(detection.get("pa_f1") or 0.0)
    seg_recall = float(detection.get("segment_recall") or 0.0)
    f1 = float(detection.get("f1") or 0.0)

    # Latency: full credit at zero delay, no credit at 48 buckets or worse.
    latency_pts = detection.get("mean_detection_latency_points")
    latency_score = (
        max(0.0, 1.0 - float(latency_pts) / 48.0) if latency_pts is not None else 0.0
    )

    # Alert burden: full credit at <= 1 false alarm/day, none at >= 10.
    fa = float(detection.get("false_alarms_per_day") or 0.0)
    burden_score = float(np.clip((10.0 - fa) / 9.0, 0.0, 1.0))

    score = (
        RANK_WEIGHTS["pa_f1"] * pa_f1
        + RANK_WEIGHTS["segment_recall"] * seg_recall
        + RANK_WEIGHTS["f1"] * f1
        + RANK_WEIGHTS["latency"] * latency_score
        + RANK_WEIGHTS["alert_burden"] * burden_score
    )
    return round(float(score), 4)


def rank_models(results: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Order model results best-first, tolerating unlabelled runs."""
    labelled = [r for r in results if r.get("composite_score") is not None]
    unlabelled = [r for r in results if r.get("composite_score") is None]

    labelled.sort(key=lambda r: r["composite_score"], reverse=True)
    # Without labels, prefer the most accurate forecaster, then the quietest.
    unlabelled.sort(
        key=lambda r: (
            (r.get("forecast") or {}).get("rmse") or float("inf"),
            (r.get("detection") or {}).get("flag_rate") or 0.0,
        )
    )

    ordered = labelled + unlabelled
    for i, row in enumerate(ordered, start=1):
        row["rank"] = i
    return ordered
