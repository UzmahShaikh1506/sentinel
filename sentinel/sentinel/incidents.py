"""
SENTINEL-TS :: Incident synthesis and decision support.

A boolean flag per bucket is not an operational product. This module turns the
winning model's flag vector into the artefact an on-call engineer can act on:
contiguous **incidents**, each with a severity, a classified shape, a
quantified deviation from expectation, and a recommended next step.

Classification is rule-based and deliberately transparent. An engineer being
paged at 03:00 needs to know *why* within seconds, and a short list of audited
rules is defensible in a way a second black-box classifier would not be. Each
rule maps to a different real cause:

``spike`` / ``dip``
    Short, sharp, returns to baseline. Usually a burst, a retry storm, a
    cache flush or a transient dependency failure.
``level_shift``
    Sustained displacement that does not revert. A deployment, a config
    change, a failover, or traffic permanently rerouted.
``ramp``
    Monotone drift away from expectation. The signature of a resource leak or
    of organic growth outpacing capacity.
``volatility``
    The mean is intact but the variance is not. Instability, contention, or a
    flapping dependency.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import SEVERITY_BANDS, SLO_THRESHOLDS


def _severity(normalised_score: float) -> str:
    for cutoff, label in SEVERITY_BANDS:
        if normalised_score >= cutoff:
            return label
    return "low"


def _classify(
    actual: np.ndarray,
    expected: Optional[np.ndarray],
    before: np.ndarray,
    after: np.ndarray,
    duration: int,
) -> Dict[str, object]:
    """Assign an incident shape from its own statistics and its surroundings."""
    if expected is not None and len(expected) == len(actual):
        deviation = actual - expected
    else:
        baseline = float(np.median(before)) if before.size else float(np.median(actual))
        deviation = actual - baseline

    mean_dev = float(np.mean(deviation))
    direction = "elevated" if mean_dev >= 0 else "suppressed"

    # Sustained displacement that persists after the window closes.
    level_shifted = False
    if before.size >= 4 and after.size >= 4:
        pre, post = float(np.median(before)), float(np.median(after))
        spread = float(np.std(before)) if np.std(before) > 1e-9 else 1.0
        level_shifted = abs(post - pre) > 2.0 * spread

    # Monotone drift inside the window.
    ramp = False
    if len(actual) >= 6:
        x = np.arange(len(actual), dtype="float64")
        slope = float(np.polyfit(x, actual, 1)[0])
        span = float(np.ptp(actual))
        ramp = span > 1e-9 and abs(slope) * len(actual) > 0.6 * span

    # Variance change with an intact mean.
    volatility = False
    if before.size >= 8:
        pre_std = float(np.std(before))
        in_std = float(np.std(actual))
        volatility = (
            pre_std > 1e-9
            and in_std > 2.5 * pre_std
            and abs(mean_dev) < 1.5 * pre_std
        )

    if volatility:
        kind = "volatility"
    elif level_shifted:
        kind = "level_shift"
    elif ramp:
        kind = "ramp"
    elif duration <= 4:
        kind = "spike" if mean_dev >= 0 else "dip"
    else:
        kind = "sustained_excursion"

    return {"kind": kind, "direction": direction, "mean_deviation": mean_dev}


#: Operator-facing guidance per incident shape. Phrased as the first checks to
#: run, not as a diagnosis -- the system observes one metric and cannot know
#: the cause.
PLAYBOOK: Dict[str, Dict[str, str]] = {
    "spike": {
        "summary": "Short, sharp excursion that returned to baseline.",
        "action": (
            "Correlate with deploy and release events in the same minutes, then "
            "with upstream retry and timeout counters. If nothing correlates and "
            "the pattern recurs, raise the debounce window rather than the "
            "threshold so genuine sustained events are still caught."
        ),
    },
    "dip": {
        "summary": "Short, sharp collapse below the expected level.",
        "action": (
            "Check for an upstream outage, a failed health check pulling the "
            "node out of rotation, or an agent that stopped reporting. A dip to "
            "exactly zero usually means missing telemetry, not idle capacity."
        ),
    },
    "level_shift": {
        "summary": "Sustained displacement with no return to the prior baseline.",
        "action": (
            "Treat as a change event until proven otherwise: inspect the deploy "
            "timeline, configuration changes, autoscaling actions and failovers "
            "at the window start. If the new level is intended, retrain the "
            "baseline so it stops paging."
        ),
    },
    "ramp": {
        "summary": "Monotone drift away from the expected trajectory.",
        "action": (
            "The classic resource-leak signature. Inspect memory and connection "
            "pools, open file handles and queue depth. Use the forward forecast "
            "to estimate when the service ceiling is reached and schedule the "
            "intervention before it, not after."
        ),
    },
    "volatility": {
        "summary": "Variance changed sharply while the mean held.",
        "action": (
            "Look for contention rather than load: noisy neighbours, lock or "
            "GC pauses, a flapping dependency, or an unhealthy member of the "
            "pool. Mean-based alerting will not catch a recurrence; keep the "
            "shape channel enabled."
        ),
    },
    "sustained_excursion": {
        "summary": "Extended departure from the expected band.",
        "action": (
            "Confirm against the forecast band and check whether demand is "
            "genuinely above plan. If it is, this is a capacity decision rather "
            "than an incident; if it is not, work upstream through the "
            "dependency chain."
        ),
    },
}


def build_incidents(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    flags: np.ndarray,
    scores: np.ndarray,
    predictions: Optional[np.ndarray] = None,
    unit: str = "units",
    family: str = "generic",
    truth: Optional[np.ndarray] = None,
    context: int = 8,
) -> List[Dict[str, object]]:
    """Collapse a flag vector into a ranked list of actionable incidents."""
    flags = np.asarray(flags, dtype=bool)
    actual = np.asarray(actual, dtype="float64")
    scores = np.asarray(scores, dtype="float64")

    if not flags.any():
        return []

    score_max = float(scores.max()) if scores.size else 1.0
    score_max = score_max if score_max > 1e-9 else 1.0

    incidents: List[Dict[str, object]] = []
    start: Optional[int] = None

    bucket_minutes = (
        float((index[1] - index[0]).total_seconds() / 60.0) if len(index) > 1 else 0.0
    )

    for i in range(len(flags) + 1):
        active = flags[i] if i < len(flags) else False
        if active and start is None:
            start = i
        elif not active and start is not None:
            end = i - 1
            incidents.append(
                _one_incident(
                    index,
                    actual,
                    scores,
                    predictions,
                    start,
                    end,
                    score_max,
                    unit,
                    family,
                    truth,
                    context,
                    bucket_minutes,
                )
            )
            start = None

    incidents.sort(key=lambda inc: inc["peak_score"], reverse=True)
    for rank, inc in enumerate(incidents, start=1):
        inc["rank"] = rank
    return incidents


def _one_incident(
    index: pd.DatetimeIndex,
    actual: np.ndarray,
    scores: np.ndarray,
    predictions: Optional[np.ndarray],
    start: int,
    end: int,
    score_max: float,
    unit: str,
    family: str,
    truth: Optional[np.ndarray],
    context: int,
    bucket_minutes: float,
) -> Dict[str, object]:
    window = slice(start, end + 1)
    seg_actual = actual[window]
    seg_scores = scores[window]
    seg_pred = predictions[window] if predictions is not None else None

    before = actual[max(0, start - context) : start]
    after = actual[end + 1 : end + 1 + context]
    duration = end - start + 1

    peak_score = float(seg_scores.max())
    normalised = float(np.clip(peak_score / score_max, 0.0, 1.0))
    severity = _severity(normalised)

    classification = _classify(seg_actual, seg_pred, before, after, duration)
    kind = classification["kind"]

    # The "peak" of an incident is where observation departs furthest from
    # what was expected, not simply where the metric was highest: a dip is as
    # much an incident as a spike.
    if seg_pred is not None:
        reference = seg_pred
    elif before.size:
        reference = np.full_like(seg_actual, float(np.median(before)))
    else:
        reference = np.full_like(seg_actual, float(np.median(seg_actual)))

    peak_idx = int(np.argmax(np.abs(seg_actual - reference)))
    peak_value = float(seg_actual[peak_idx])
    expected_at_peak = float(reference[peak_idx])
    deviation_pct = (
        100.0 * (peak_value - expected_at_peak) / abs(expected_at_peak)
        if abs(expected_at_peak) > 1e-9
        else None
    )

    slo = SLO_THRESHOLDS.get(family, SLO_THRESHOLDS["generic"])
    ceiling = slo.get("ceiling")
    breached = bool(ceiling is not None and float(seg_actual.max()) >= ceiling)

    confirmed = None
    if truth is not None:
        confirmed = bool(np.asarray(truth, dtype=bool)[window].any())

    play = PLAYBOOK.get(kind, PLAYBOOK["sustained_excursion"])

    return {
        "start": index[start].isoformat(),
        "end": index[end].isoformat(),
        "start_index": int(start),
        "end_index": int(end),
        "duration_buckets": int(duration),
        "duration_minutes": round(duration * bucket_minutes, 1),
        "peak_score": round(peak_score, 4),
        "normalised_score": round(normalised, 4),
        "severity": severity,
        "kind": kind,
        "direction": classification["direction"],
        "peak_value": round(peak_value, 4),
        "expected_value": round(expected_at_peak, 4),
        "deviation": round(peak_value - expected_at_peak, 4),
        "deviation_pct": round(deviation_pct, 1) if deviation_pct is not None else None,
        "mean_value": round(float(seg_actual.mean()), 4),
        "unit": unit,
        "slo_ceiling": ceiling,
        "slo_breached": breached,
        "confirmed_by_ground_truth": confirmed,
        "summary": play["summary"],
        "recommended_action": play["action"],
        "headline": _headline(kind, deviation_pct, duration, bucket_minutes, unit, peak_value),
    }


def _headline(
    kind: str,
    deviation_pct: Optional[float],
    duration: int,
    bucket_minutes: float,
    unit: str,
    peak_value: float,
) -> str:
    minutes = duration * bucket_minutes
    span = f"{minutes:.0f} min" if minutes >= 1 else f"{duration} buckets"
    label = {
        "spike": "Spike",
        "dip": "Dip",
        "level_shift": "Level shift",
        "ramp": "Ramp",
        "volatility": "Volatility change",
        "sustained_excursion": "Sustained excursion",
    }.get(kind, "Anomaly")

    if deviation_pct is not None and abs(deviation_pct) >= 1:
        return (
            f"{label}: peak {peak_value:,.2f} {unit}, "
            f"{deviation_pct:+.0f}% against forecast, sustained {span}."
        )
    return f"{label}: peak {peak_value:,.2f} {unit}, sustained {span}."


# --------------------------------------------------------------------------
# Capacity decision support
# --------------------------------------------------------------------------


def capacity_outlook(
    last_timestamp: pd.Timestamp,
    bucket: pd.Timedelta,
    forecast: Optional[Dict[str, object]],
    family: str,
    recent: np.ndarray,
) -> Dict[str, object]:
    """Forward-looking guidance derived from the winning model's forecast.

    Where a service-level ceiling exists for the metric family, the first
    forecast bucket expected to breach it is reported along with its upper
    prediction bound, so the operator can act on the pessimistic case rather
    than the central one.
    """
    slo = SLO_THRESHOLDS.get(family, SLO_THRESHOLDS["generic"])
    ceiling = slo.get("ceiling")
    out: Dict[str, object] = {
        "available": forecast is not None,
        "ceiling": ceiling,
        "unit": slo.get("unit", "units"),
    }
    if forecast is None:
        out["message"] = "The selected model does not produce a forward forecast."
        return out

    values = np.asarray(forecast.get("values", []), dtype="float64")
    upper = forecast.get("upper")
    upper_arr = np.asarray(upper, dtype="float64") if upper else None

    timestamps = [
        (last_timestamp + bucket * (i + 1)).isoformat() for i in range(len(values))
    ]
    out["timestamps"] = timestamps
    out["values"] = [round(float(v), 4) for v in values]
    if upper_arr is not None:
        out["upper"] = [round(float(v), 4) for v in upper_arr]
        out["lower"] = [round(float(v), 4) for v in np.asarray(forecast["lower"])]

    out["horizon_buckets"] = int(len(values))
    out["method"] = forecast.get("method", "")

    if values.size:
        out["forecast_peak"] = round(float(values.max()), 4)
        out["forecast_mean"] = round(float(values.mean()), 4)
        baseline = float(np.median(recent)) if recent.size else None
        if baseline is not None and abs(baseline) > 1e-9:
            out["expected_change_pct"] = round(
                100.0 * (float(values.mean()) - baseline) / abs(baseline), 1
            )

    if ceiling is None:
        out["message"] = (
            "No fixed service ceiling is defined for this metric family, so the "
            "forecast is advisory: compare it against the capacity you have "
            "provisioned rather than against an absolute limit."
        )
        return out

    series_to_test = upper_arr if upper_arr is not None else values
    breach = np.where(series_to_test >= ceiling)[0]
    if breach.size:
        idx = int(breach[0])
        out["breach_expected"] = True
        out["breach_index"] = idx
        out["breach_timestamp"] = timestamps[idx]
        out["breach_in_minutes"] = round(
            (idx + 1) * bucket.total_seconds() / 60.0, 1
        )
        out["message"] = (
            f"The upper 95% forecast bound crosses the {ceiling}"
            f"{slo.get('unit','')} ceiling about "
            f"{out['breach_in_minutes']:.0f} minutes from the end of the data. "
            "Treat this as a scheduling signal for headroom, not as a prediction "
            "that the breach will certainly occur."
        )
    else:
        out["breach_expected"] = False
        out["message"] = (
            f"The forecast stays below the {ceiling}{slo.get('unit','')} ceiling "
            "across the whole horizon, including the upper prediction bound. No "
            "capacity action is indicated from this metric alone."
        )
    return out


def alert_digest(incidents: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Counts and a one-line verdict for the alert feed header."""
    by_severity = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    by_kind: Dict[str, int] = {}
    for inc in incidents:
        by_severity[inc["severity"]] = by_severity.get(inc["severity"], 0) + 1
        by_kind[inc["kind"]] = by_kind.get(inc["kind"], 0) + 1

    total = len(incidents)
    actionable = by_severity["critical"] + by_severity["high"]

    if total == 0:
        verdict = "No incidents detected in the evaluation window."
    elif actionable == 0:
        verdict = (
            f"{total} low-severity deviation(s) detected; none meet the bar for "
            "paging and all can be reviewed in business hours."
        )
    else:
        verdict = (
            f"{actionable} incident(s) at high or critical severity out of "
            f"{total} detected; these are the ones worth waking someone for."
        )

    return {
        "total": total,
        "by_severity": by_severity,
        "by_kind": by_kind,
        "actionable": actionable,
        "verdict": verdict,
    }
