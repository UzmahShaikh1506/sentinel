"""
SENTINEL-TS :: Dataset catalogue and loading.

The bundled corpus is the ``realAWSCloudwatch`` family of the Numenta Anomaly
Benchmark (NAB) -- genuine Amazon CloudWatch server telemetry sampled every
five minutes, shipped with human-curated anomaly *windows* rather than point
labels. Window labels are the realistic framing for SRE work: an operator
cares that the incident was caught inside its window, not that a particular
five-minute bucket was flagged.

This module is the only place that knows where bytes live on disk. Everything
downstream consumes a tidy ``pandas.Series`` plus a list of label windows.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .config import LABEL_FILE, RAW_DIR, UPLOAD_DIR

# --------------------------------------------------------------------------
# Catalogue metadata
# --------------------------------------------------------------------------

#: Human-facing descriptions for each bundled series. ``family`` drives the
#: decision-support thresholds; ``service`` is the fictional-but-plausible
#: service name shown in the console so the demo reads like a real estate of
#: infrastructure rather than a list of file names.
CATALOGUE: Dict[str, Dict[str, str]] = {
    "ec2_cpu_utilization_5f5533": {
        "title": "EC2 CPU Utilisation",
        "service": "api-gateway-prod-01",
        "family": "cpu",
        "unit": "%",
        "description": (
            "CPU utilisation of a production EC2 instance fronting the public "
            "API. Contains two curated anomaly windows."
        ),
    },
    "rds_cpu_utilization_cc0c53": {
        "title": "RDS CPU Utilisation",
        "service": "orders-db-primary",
        "family": "cpu",
        "unit": "%",
        "description": (
            "CPU utilisation of the primary relational database. Anomalies "
            "correspond to sustained load excursions."
        ),
    },
    "ec2_network_in_257a54": {
        "title": "EC2 Network Ingress",
        "service": "edge-ingest-02",
        "family": "network",
        "unit": "bytes",
        "description": (
            "Inbound network volume on an ingest node. A single long anomaly "
            "window marks an abnormal traffic regime."
        ),
    },
    "elb_request_count_8c0756": {
        "title": "ELB Request Count",
        "service": "public-lb-eu-west",
        "family": "requests",
        "unit": "req/bucket",
        "description": (
            "Request volume reaching the public load balancer. Two anomaly "
            "windows capture demand shocks."
        ),
    },
    "ec2_disk_write_bytes_1ef3de": {
        "title": "EC2 Disk Write Bytes",
        "service": "batch-worker-07",
        "family": "disk",
        "unit": "bytes",
        "description": (
            "Disk write throughput on a batch worker. The anomaly window marks "
            "a runaway write regime."
        ),
    },
    "grok_asg_anomaly": {
        "title": "Auto-Scaling Group Load",
        "service": "asg-checkout-fleet",
        "family": "generic",
        "unit": "units",
        "description": (
            "Aggregate load across an auto-scaling group, with three curated "
            "anomaly windows."
        ),
    },
}

_UPLOAD_PREFIX = "upload::"


@dataclass
class LoadedSeries:
    """A raw series plus everything known about its provenance."""

    dataset_id: str
    series: pd.Series
    windows: List[Tuple[pd.Timestamp, pd.Timestamp]]
    meta: Dict[str, object]

    @property
    def n_obs(self) -> int:
        return int(len(self.series))


# --------------------------------------------------------------------------
# Label handling
# --------------------------------------------------------------------------


def _load_label_file() -> Dict[str, List[List[str]]]:
    if not LABEL_FILE.exists():
        return {}
    with open(LABEL_FILE, "r", encoding="utf-8") as fh:
        return json.load(fh)


def label_windows_for(dataset_id: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return ground-truth anomaly windows for a bundled dataset.

    NAB keys windows by their path inside the benchmark (for example
    ``realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv``), so the lookup matches
    on the file stem.
    """
    labels = _load_label_file()
    for key, windows in labels.items():
        if Path(key).stem == dataset_id:
            return [
                (pd.Timestamp(start), pd.Timestamp(end)) for start, end in windows
            ]
    return []


# --------------------------------------------------------------------------
# Catalogue API
# --------------------------------------------------------------------------


def thumbnail(series: pd.Series, points: int = 64) -> List[float]:
    """Heavily thinned copy of a series for the picker's sparklines."""
    step = max(1, len(series) // points)
    return [round(float(v), 4) for v in series.iloc[::step].to_numpy()[:points]]


def preview(dataset_id: str, points: int = 900) -> Dict[str, object]:
    """Thinned raw series for the console's pre-run preview chart."""
    loaded = load_dataset(dataset_id)
    series = loaded.series
    step = max(1, len(series) // points)
    thin = series.iloc[::step]
    return {
        "id": dataset_id,
        "timestamps": [ts.isoformat() for ts in thin.index],
        "values": [round(float(v), 4) for v in thin.to_numpy()],
        "thinning_step": step,
        "label_windows": [
            {"start": s.isoformat(), "end": e.isoformat()} for s, e in loaded.windows
        ],
        "meta": loaded.meta,
    }


def list_datasets() -> List[Dict[str, object]]:
    """Everything the console needs to render the dataset picker."""
    out: List[Dict[str, object]] = []

    for dataset_id, meta in CATALOGUE.items():
        path = RAW_DIR / f"{dataset_id}.csv"
        if not path.exists():
            continue
        try:
            series = _read_csv_series(path)
        except Exception:  # pragma: no cover - corrupt file guard
            continue
        windows = label_windows_for(dataset_id)
        out.append(
            {
                "id": dataset_id,
                "title": meta["title"],
                "service": meta["service"],
                "family": meta["family"],
                "unit": meta["unit"],
                "description": meta["description"],
                "n_obs": int(len(series)),
                "start": series.index[0].isoformat(),
                "end": series.index[-1].isoformat(),
                "native_freq": "5min",
                "n_label_windows": len(windows),
                "source": "Numenta Anomaly Benchmark - realAWSCloudwatch",
                "spark": thumbnail(series),
                "builtin": True,
            }
        )

    for path in sorted(UPLOAD_DIR.glob("*.csv")):
        try:
            series = _read_csv_series(path)
        except Exception:
            continue
        out.append(
            {
                "id": _UPLOAD_PREFIX + path.stem,
                "title": path.stem.replace("_", " ").title(),
                "service": "user-upload",
                "family": "generic",
                "unit": "units",
                "description": "User-supplied time series uploaded through the console.",
                "n_obs": int(len(series)),
                "start": series.index[0].isoformat(),
                "end": series.index[-1].isoformat(),
                "native_freq": _infer_freq_label(series.index),
                "n_label_windows": 0,
                "source": "User upload",
                "spark": thumbnail(series),
                "builtin": False,
            }
        )

    return out


def dataset_meta(dataset_id: str) -> Dict[str, object]:
    if dataset_id.startswith(_UPLOAD_PREFIX):
        stem = dataset_id[len(_UPLOAD_PREFIX) :]
        return {
            "title": stem.replace("_", " ").title(),
            "service": "user-upload",
            "family": "generic",
            "unit": "units",
            "description": "User-supplied time series.",
        }
    if dataset_id not in CATALOGUE:
        raise KeyError(f"Unknown dataset: {dataset_id}")
    return dict(CATALOGUE[dataset_id])


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

_TIME_CANDIDATES = ("timestamp", "time", "datetime", "date", "ts")
_VALUE_CANDIDATES = ("value", "val", "metric", "y", "reading", "count")


def _pick_column(columns: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    lowered = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _read_csv_series(path: Path) -> pd.Series:
    """Read a two-column (timestamp, value) CSV into a sorted float Series.

    Column names are discovered rather than assumed, so a user upload that
    calls its columns ``date``/``reading`` loads without edits. If discovery
    fails the first two columns are used positionally.
    """
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(
            f"{path.name}: expected at least two columns (timestamp, value), "
            f"found {frame.shape[1]}."
        )

    cols = list(frame.columns.astype(str))
    time_col = _pick_column(cols, _TIME_CANDIDATES) or cols[0]
    value_col = _pick_column(cols, _VALUE_CANDIDATES)
    if value_col is None or value_col == time_col:
        remaining = [c for c in cols if c != time_col]
        value_col = remaining[0]

    index = pd.to_datetime(frame[time_col], errors="coerce", format="mixed")
    values = pd.to_numeric(frame[value_col], errors="coerce")

    series = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(index), name="value")
    series = series[~series.index.isna()]
    series = series.sort_index()
    series = series.astype("float64")
    if series.empty:
        raise ValueError(f"{path.name}: no parseable rows found.")
    return series


def _infer_freq_label(index: pd.DatetimeIndex) -> str:
    if len(index) < 3:
        return "unknown"
    deltas = pd.Series(index[1:]) - pd.Series(index[:-1])
    median = deltas.median()
    total_seconds = median.total_seconds()
    if total_seconds % 3600 == 0:
        return f"{int(total_seconds // 3600)}h"
    if total_seconds % 60 == 0:
        return f"{int(total_seconds // 60)}min"
    return f"{int(total_seconds)}s"


def resolve_path(dataset_id: str) -> Path:
    if dataset_id.startswith(_UPLOAD_PREFIX):
        return UPLOAD_DIR / f"{dataset_id[len(_UPLOAD_PREFIX):]}.csv"
    return RAW_DIR / f"{dataset_id}.csv"


def load_dataset(dataset_id: str) -> LoadedSeries:
    """Load a catalogue or uploaded series together with its labels."""
    path = resolve_path(dataset_id)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found for id '{dataset_id}'.")

    series = _read_csv_series(path)
    windows = (
        [] if dataset_id.startswith(_UPLOAD_PREFIX) else label_windows_for(dataset_id)
    )
    meta = dataset_meta(dataset_id)
    meta.update(
        {
            "id": dataset_id,
            "native_freq": _infer_freq_label(series.index),
            "n_obs": int(len(series)),
            "start": series.index[0].isoformat(),
            "end": series.index[-1].isoformat(),
            "n_label_windows": len(windows),
            "has_labels": bool(windows),
        }
    )
    return LoadedSeries(dataset_id=dataset_id, series=series, windows=windows, meta=meta)


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def save_upload(filename: str, content: bytes) -> str:
    """Persist an uploaded CSV and return its dataset id.

    The stored name is sanitised to a conservative character set so that a
    hostile filename cannot escape the upload directory.
    """
    stem = Path(filename).stem or "uploaded_series"
    stem = _SAFE_NAME.sub("_", stem)[:64] or "uploaded_series"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = UPLOAD_DIR / f"{stem}_{stamp}.csv"
    target.write_bytes(content)

    try:
        _read_csv_series(target)  # validate before advertising it
    except Exception:
        os.unlink(target)
        raise

    return _UPLOAD_PREFIX + target.stem
