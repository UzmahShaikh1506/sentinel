"""
SENTINEL-TS :: Orchestration.

One function, :func:`run_experiment`, carries a dataset from disk to a fully
serialisable result bundle: preprocessing audit, EDA, a scored model
comparison, the winning model's incidents, and the forward capacity outlook.

The whole bundle is JSON-ready. The API layer adds no analysis of its own --
it only routes - which keeps the science testable without a web server
running.
"""

from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import RunConfig
from .datasets import load_dataset
from .eda import run_eda
from .incidents import alert_digest, build_incidents, capacity_outlook
from .metrics import composite_score, detection_metrics, forecast_metrics, rank_models
from .models import REGISTRY, normalise_scores
from .preprocessing import preprocess, train_test_split, windows_to_point_labels

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def _f(values) -> List[Optional[float]]:
    out: List[Optional[float]] = []
    for v in np.asarray(values, dtype="float64"):
        out.append(None if not np.isfinite(v) else round(float(v), 4))
    return out


def run_experiment(config: RunConfig, progress=None) -> Dict[str, object]:
    """Execute the full analysis for one dataset and configuration.

    Parameters
    ----------
    config
        Dataset, resampling frequency, split, thresholds and model selection.
    progress
        Optional ``callable(stage: str, pct: int)`` used by the console to
        drive its pipeline animation.
    """
    started = time.perf_counter()

    def emit(stage: str, pct: int) -> None:
        if progress is not None:
            progress(stage, pct)

    # ---------------- 1. Ingest -------------------------------------------
    emit("Loading telemetry", 5)
    loaded = load_dataset(config.dataset_id)

    # ---------------- 2. Preprocess ---------------------------------------
    emit("Cleaning and resampling", 15)
    clean, outlier_flags, report = preprocess(loaded.series, freq=config.freq)
    period = config.seasonal_period

    # ---------------- 3. EDA ----------------------------------------------
    emit("Exploratory analysis", 28)
    eda = run_eda(clean, period)

    suggested_order = (
        int(eda["correlation"]["suggested_p"]),
        int(eda["stationarity"]["verdict"]["recommended_d"]),
        int(eda["correlation"]["suggested_q"]),
    )

    # ---------------- 4. Split --------------------------------------------
    emit("Splitting train and test", 34)
    train, test = train_test_split(clean, config.test_size)

    truth = (
        windows_to_point_labels(test.index, loaded.windows)
        if loaded.windows
        else None
    )
    train_truth = (
        windows_to_point_labels(train.index, loaded.windows)
        if loaded.windows
        else None
    )

    # ---------------- 5. Models -------------------------------------------
    selected = [m for m in config.models if m in REGISTRY]
    if not selected:
        selected = list(REGISTRY.keys())

    results: List[Dict[str, object]] = []
    raw_outputs: Dict[str, object] = {}

    span = 50
    for i, name in enumerate(selected):
        pct = 36 + int(span * i / max(len(selected), 1))
        cls = REGISTRY[name]
        emit(f"Fitting {cls.display_name}", pct)

        detector = cls(
            period=period,
            threshold_sigma=config.threshold_sigma,
            rolling_window=config.rolling_window,
            contamination=config.contamination,
            fourier_k=config.fourier_k,
            suggested_order=suggested_order,
            horizon=config.horizon,
        )
        output = detector.run(train, test)
        raw_outputs[name] = output

        forecast = None
        if output.predictions is not None and output.error is None:
            forecast = forecast_metrics(
                test.to_numpy(dtype="float64"),
                output.predictions,
                insample=train.to_numpy(dtype="float64"),
                seasonal_period=period,
            )

        detection = detection_metrics(output.flags, truth, test.index)
        score = composite_score(detection, forecast)

        results.append(
            {
                "name": output.name,
                "display_name": output.display_name,
                "family": output.family,
                "description": output.description,
                "rationale": output.rationale,
                "forecast": forecast,
                "detection": detection,
                "composite_score": score,
                "threshold": round(float(output.threshold), 4),
                "fit_seconds": output.fit_seconds,
                "params": output.params,
                "diagnostics": output.diagnostics,
                "error": output.error,
            }
        )

    emit("Scoring and ranking", 88)
    ranked = rank_models(results)

    # ---------------- 6. Winner and incidents ------------------------------
    winner_name = ranked[0]["name"] if ranked else selected[0]
    winner = raw_outputs[winner_name]
    meta = loaded.meta

    emit("Building incident report", 93)
    incidents = build_incidents(
        index=test.index,
        actual=test.to_numpy(dtype="float64"),
        flags=winner.flags,
        scores=winner.scores,
        predictions=winner.predictions,
        unit=str(meta.get("unit", "units")),
        family=str(meta.get("family", "generic")),
        truth=truth,
    )
    digest = alert_digest(incidents)

    bucket = test.index[1] - test.index[0] if len(test.index) > 1 else pd.Timedelta(config.freq)
    outlook = capacity_outlook(
        last_timestamp=clean.index[-1],
        bucket=bucket,
        forecast=winner.forecast_future,
        family=str(meta.get("family", "generic")),
        recent=test.to_numpy(dtype="float64")[-period:],
    )

    # ---------------- 7. Serialise ----------------------------------------
    emit("Finalising", 97)

    series_payload = {
        "train": {
            "timestamps": [ts.isoformat() for ts in train.index],
            "values": _f(train.to_numpy()),
        },
        "test": {
            "timestamps": [ts.isoformat() for ts in test.index],
            "values": _f(test.to_numpy()),
        },
        "outliers_flagged": [
            ts.isoformat() for ts in clean.index[outlier_flags.to_numpy()]
        ],
        "truth_mask": truth.tolist() if truth is not None else None,
        "train_truth_mask": train_truth.tolist() if train_truth is not None else None,
        "label_windows": [
            {"start": s.isoformat(), "end": e.isoformat()} for s, e in loaded.windows
        ],
    }

    model_series: Dict[str, object] = {}
    for name, output in raw_outputs.items():
        model_series[name] = {
            "predictions": _f(output.predictions) if output.predictions is not None else None,
            "lower": _f(output.lower) if output.lower is not None else None,
            "upper": _f(output.upper) if output.upper is not None else None,
            "scores": _f(output.scores),
            "normalised_scores": _f(normalise_scores(output.scores)),
            "flags": [bool(x) for x in output.flags],
            "threshold": round(float(output.threshold), 4),
            "forecast_future": output.forecast_future,
        }

    elapsed = time.perf_counter() - started
    emit("Complete", 100)

    return {
        "config": config.to_dict(),
        "dataset": meta,
        "preprocessing": report.to_dict(),
        "eda": eda,
        "split": {
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "train_start": train.index[0].isoformat(),
            "train_end": train.index[-1].isoformat(),
            "test_start": test.index[0].isoformat(),
            "test_end": test.index[-1].isoformat(),
            "method": "chronological hold-out (no shuffling)",
            "rationale": (
                "Random splitting would leak future information through the "
                "series' own autocorrelation, so the hold-out is strictly the "
                "tail of the timeline."
            ),
        },
        "suggested_order_from_eda": list(suggested_order),
        "models": ranked,
        "model_series": model_series,
        "series": series_payload,
        "winner": winner_name,
        "incidents": incidents,
        "digest": digest,
        "capacity_outlook": outlook,
        "runtime_seconds": round(elapsed, 2),
        "narrative": _narrative(ranked, digest, loaded.meta, truth is not None),
    }


def _narrative(
    ranked: List[Dict[str, object]],
    digest: Dict[str, object],
    meta: Dict[str, object],
    labelled: bool,
) -> Dict[str, object]:
    """Plain-language conclusions for the console's summary panel."""
    if not ranked:
        return {"headline": "No models completed.", "points": []}

    best = ranked[0]
    points: List[str] = []

    if labelled and best["composite_score"] is not None:
        det = best["detection"]
        points.append(
            f"{best['display_name']} ranks first with a composite operational "
            f"score of {best['composite_score']:.3f}, recovering "
            f"{det['n_segments_detected']}/{det['n_true_segments']} labelled "
            f"incident window(s) at a point-adjusted F1 of {det['pa_f1']:.3f}."
        )
        if det.get("mean_detection_latency_minutes") is not None:
            points.append(
                f"Mean time to first alert inside an incident is "
                f"{det['mean_detection_latency_minutes']:.0f} minutes, at "
                f"{det['false_alarms_per_day']:.2f} false alarm(s) per day."
            )
    else:
        points.append(
            f"{best['display_name']} ranks first on forecast accuracy. This "
            "series carries no ground-truth labels, so detection quality cannot "
            "be scored and the ranking falls back to RMSE and alert volume."
        )

    # Compare against the honest baseline rather than only celebrating the winner.
    baseline = next((r for r in ranked if r["name"] == "seasonal_naive"), None)
    if baseline and baseline is not best and baseline.get("composite_score") is not None:
        delta = (best["composite_score"] or 0) - baseline["composite_score"]
        if delta <= 0.02:
            points.append(
                "The margin over the seasonal-naive baseline is slight. On this "
                "series the added complexity is not clearly earning its keep, "
                "which is worth stating plainly rather than hiding in the table."
            )
        else:
            points.append(
                f"That is {delta:+.3f} against the seasonal-naive baseline, so "
                "the modelling effort is justified on this series."
            )

    hybrids = [r for r in ranked if r["family"] == "hybrid" and not r.get("error")]
    if hybrids:
        top_hybrid = hybrids[0]
        reduction = (top_hybrid.get("diagnostics") or {}).get(
            "residual_variance_reduction_pct"
        )
        if reduction is not None:
            points.append(
                f"The nonlinear stage removed {reduction:.1f}% of the variance "
                "remaining in the linear model's residuals, which is the direct "
                "measure of what hybridisation bought."
            )

    points.append(digest["verdict"])

    return {
        "headline": (
            f"{best['display_name']} selected for {meta.get('service', 'this service')}"
        ),
        "points": points,
    }
