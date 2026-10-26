"""
SENTINEL-TS :: Cross-dataset benchmark.

A single NAB series contains only one or two labelled incident windows inside
a chronological hold-out. That is enough to demonstrate the pipeline but far
too thin to rank models on: with one window, segment recall is either 0 or 1
and the leaderboard turns on a handful of points.

This module removes that weakness by running the identical pipeline across
every labelled series in the corpus and aggregating. Two aggregations are
reported because they answer different questions:

**Macro average**
    Mean of each model's per-dataset score. Treats every service as equally
    important, which is the fair comparison when the series differ in length.

**Pooled (micro) counts**
    True/false positives and negatives summed across datasets, then scored
    once. Dominated by the longer series, and the better estimate of what the
    detector would do across a fleet.

A model that wins on one but not the other is unstable, and the summary says
so rather than quietly picking the flattering number.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import RunConfig
from .datasets import CATALOGUE, list_datasets
from .metrics import _prf
from .pipeline import run_experiment


def labelled_dataset_ids() -> List[str]:
    """Catalogue datasets that ship ground-truth windows."""
    return [
        d["id"]
        for d in list_datasets()
        if d.get("builtin") and d.get("n_label_windows", 0) > 0
    ]


def run_benchmark(
    dataset_ids: Optional[List[str]] = None,
    base_config: Optional[RunConfig] = None,
    progress=None,
) -> Dict[str, object]:
    """Run every model over every dataset and aggregate the scorecards."""
    ids = dataset_ids or labelled_dataset_ids()
    template = base_config or RunConfig(dataset_id=ids[0])

    per_dataset: List[Dict[str, object]] = []
    accumulator: Dict[str, Dict[str, object]] = {}

    for i, dataset_id in enumerate(ids):
        if progress is not None:
            progress(
                f"Benchmarking {CATALOGUE.get(dataset_id, {}).get('title', dataset_id)}",
                int(100 * i / max(len(ids), 1)),
            )

        config = RunConfig(
            dataset_id=dataset_id,
            freq=template.freq,
            test_size=template.test_size,
            fourier_k=template.fourier_k,
            threshold_sigma=template.threshold_sigma,
            rolling_window=template.rolling_window,
            contamination=template.contamination,
            horizon=template.horizon,
            models=list(template.models),
        )

        try:
            result = run_experiment(config)
        except Exception as exc:  # pragma: no cover - one bad series must not
            per_dataset.append({"dataset": dataset_id, "error": str(exc)})
            continue

        rows = []
        for model in result["models"]:
            det = model["detection"]
            fc = model["forecast"] or {}
            rows.append(
                {
                    "name": model["name"],
                    "display_name": model["display_name"],
                    "family": model["family"],
                    "composite_score": model["composite_score"],
                    "pa_f1": det.get("pa_f1"),
                    "f1": det.get("f1"),
                    "segment_recall": det.get("segment_recall"),
                    "false_alarms_per_day": det.get("false_alarms_per_day"),
                    "latency_points": det.get("mean_detection_latency_points"),
                    "rmse": fc.get("rmse"),
                    "mase": fc.get("mase"),
                    "fit_seconds": model["fit_seconds"],
                    "error": model.get("error"),
                }
            )

            acc = accumulator.setdefault(
                model["name"],
                {
                    "name": model["name"],
                    "display_name": model["display_name"],
                    "family": model["family"],
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "tn": 0,
                    "segments_total": 0,
                    "segments_detected": 0,
                    "composite_scores": [],
                    "pa_f1s": [],
                    "latencies": [],
                    "false_alarms": [],
                    "rmses": [],
                    "mases": [],
                    "fit_seconds": [],
                    "errors": 0,
                },
            )
            if model.get("error"):
                acc["errors"] += 1
                continue

            for key in ("tp", "fp", "fn", "tn"):
                if det.get(key) is not None:
                    acc[key] += int(det[key])
            acc["segments_total"] += int(det.get("n_true_segments") or 0)
            acc["segments_detected"] += int(det.get("n_segments_detected") or 0)
            for src, dst in (
                ("composite_score", "composite_scores"),
                ("pa_f1", "pa_f1s"),
            ):
                value = model.get(src) if src == "composite_score" else det.get(src)
                if value is not None:
                    acc[dst].append(float(value))
            if det.get("mean_detection_latency_points") is not None:
                acc["latencies"].append(float(det["mean_detection_latency_points"]))
            if det.get("false_alarms_per_day") is not None:
                acc["false_alarms"].append(float(det["false_alarms_per_day"]))
            if fc.get("rmse") is not None:
                acc["rmses"].append(float(fc["rmse"]))
            if fc.get("mase") is not None:
                acc["mases"].append(float(fc["mase"]))
            acc["fit_seconds"].append(float(model["fit_seconds"]))

        per_dataset.append(
            {
                "dataset": dataset_id,
                "title": result["dataset"].get("title", dataset_id),
                "service": result["dataset"].get("service"),
                "winner": result["winner"],
                "n_test": result["split"]["n_test"],
                "n_label_windows": result["dataset"].get("n_label_windows"),
                "models": rows,
            }
        )

    summary = _summarise(accumulator, len(per_dataset))
    if progress is not None:
        progress("Complete", 100)

    return {
        "datasets": per_dataset,
        "summary": summary,
        "n_datasets": len(per_dataset),
        "config": template.to_dict(),
        "notes": _notes(summary),
    }


def _mean(values: List[float]) -> Optional[float]:
    return round(float(np.mean(values)), 4) if values else None


def _summarise(
    accumulator: Dict[str, Dict[str, object]], n_datasets: int
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for acc in accumulator.values():
        precision, recall, f1 = _prf(acc["tp"], acc["fp"], acc["fn"])
        rows.append(
            {
                "name": acc["name"],
                "display_name": acc["display_name"],
                "family": acc["family"],
                "n_datasets": n_datasets,
                "errors": acc["errors"],
                "macro_composite": _mean(acc["composite_scores"]),
                "macro_pa_f1": _mean(acc["pa_f1s"]),
                "macro_latency_points": _mean(acc["latencies"]),
                "macro_false_alarms_per_day": _mean(acc["false_alarms"]),
                # RMSE is reported per service but deliberately NOT averaged
                # across the corpus: the series are measured in percent, bytes
                # and request counts, so a mean of their RMSEs is a mean of
                # incommensurable quantities. MASE is scale-free by
                # construction and is the forecast metric that aggregates
                # honestly.
                "macro_mase": _mean(acc["mases"]),
                "rmse_not_aggregated": (
                    "RMSE is unit-dependent and is not averaged across services; "
                    "see the per-service table. MASE is the comparable measure."
                ),
                "mean_fit_seconds": _mean(acc["fit_seconds"]),
                "pooled_precision": round(precision, 4),
                "pooled_recall": round(recall, 4),
                "pooled_f1": round(f1, 4),
                "pooled_tp": acc["tp"],
                "pooled_fp": acc["fp"],
                "pooled_fn": acc["fn"],
                "segments_total": acc["segments_total"],
                "segments_detected": acc["segments_detected"],
                "segment_recall": (
                    round(acc["segments_detected"] / acc["segments_total"], 4)
                    if acc["segments_total"]
                    else None
                ),
            }
        )

    rows.sort(key=lambda r: (r["macro_composite"] or 0.0), reverse=True)
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def _notes(summary: List[Dict[str, object]]) -> List[str]:
    if not summary:
        return []

    notes: List[str] = []
    macro_best = summary[0]
    pooled_best = max(summary, key=lambda r: r["pooled_f1"])

    notes.append(
        f"Across the corpus, {macro_best['display_name']} leads on the macro "
        f"composite score ({macro_best['macro_composite']}), recovering "
        f"{macro_best['segments_detected']}/{macro_best['segments_total']} "
        "labelled incident windows."
    )

    if pooled_best["name"] != macro_best["name"]:
        notes.append(
            f"The pooled point-wise F1 instead favours "
            f"{pooled_best['display_name']} ({pooled_best['pooled_f1']}). The "
            "two measures disagreeing means no single model dominates: one is "
            "better per service, the other better per observation across the "
            "fleet. That disagreement is the honest result."
        )
    else:
        notes.append(
            "The macro and pooled rankings agree, so the leader is stable to "
            "the choice of aggregation rather than an artefact of it."
        )

    quiet = min(summary, key=lambda r: r["macro_false_alarms_per_day"] or 0.0)
    notes.append(
        f"{quiet['display_name']} is the quietest detector at "
        f"{quiet['macro_false_alarms_per_day']} false alarm(s) per day on "
        "average; alert burden should be weighed alongside recall, because a "
        "detector that is ignored has an effective recall of zero."
    )
    return notes
