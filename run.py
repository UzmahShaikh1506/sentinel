#!/usr/bin/env python3
"""
SENTINEL-TS launcher.

Usage
-----
    python run.py                    Start the console on http://127.0.0.1:8000
    python run.py --port 9000        Use a different port
    python run.py --check            Verify the environment and dataset, then exit
    python run.py --cli DATASET_ID   Run one analysis in the terminal, no server
    python run.py --benchmark        Run the cross-dataset benchmark in the terminal

The server binds to localhost only. Nothing in this project needs to be
reachable from the network, and binding to all interfaces by default would
expose an unauthenticated API.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BANNER = r"""
   ___ ___ _  _ _____ ___ _  _ ___ _      _____ ___
  / __| __| \| |_   _|_ _| \| | __| |    |_   _/ __|
  \__ \ _|| .` | | |  | || .` | _|| |__    | | \__ \
  |___/___|_|\_| |_| |___|_|\_|___|____|   |_| |___/
  Hybrid time-series anomaly intelligence for cloud telemetry
"""

REQUIRED = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("sklearn", "scikit-learn"),
    ("statsmodels", "statsmodels"),
    ("fastapi", "fastapi"),
    ("uvicorn", "uvicorn"),
]


def check_environment() -> bool:
    """Verify dependencies and bundled data before anything else runs."""
    import importlib

    ok = True
    print("Dependencies")
    for module, package in REQUIRED:
        try:
            m = importlib.import_module(module)
            print(f"  [ ok ] {package:<14} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  [MISS] {package:<14} -> pip install {package}")

    try:
        importlib.import_module("torch")
        print("  [ ok ] torch          (optional LSTM residual stage enabled)")
    except ImportError:
        print(
            "  [ -- ] torch          optional; the hybrid falls back to a "
            "feed-forward network"
        )

    print("\nBundled data")
    from sentinel.config import LABEL_FILE, RAW_DIR

    csvs = sorted(RAW_DIR.glob("*.csv"))
    if csvs:
        print(f"  [ ok ] {len(csvs)} series in {RAW_DIR}")
        for p in csvs:
            print(f"         - {p.name}")
    else:
        ok = False
        print(f"  [MISS] no CSV files in {RAW_DIR}")

    if LABEL_FILE.exists():
        print(f"  [ ok ] ground-truth labels: {LABEL_FILE.name}")
    else:
        print("  [ -- ] no label file; detection metrics will be unavailable")

    if ok:
        from sentinel.datasets import list_datasets

        print(f"\n  {len(list_datasets())} dataset(s) registered and readable.")
    print()
    return ok


def cli_run(dataset_id: str, freq: str, test_size: float) -> int:
    from sentinel import RunConfig, run_experiment

    config = RunConfig(dataset_id=dataset_id, freq=freq, test_size=test_size)

    def progress(stage: str, pct: int) -> None:
        bar = "#" * (pct // 3)
        sys.stdout.write(f"\r  [{bar:<34}] {pct:3d}%  {stage[:38]:<38}")
        sys.stdout.flush()

    print(f"Analysing {dataset_id} at {freq} buckets, {test_size:.0%} hold-out\n")
    result = run_experiment(config, progress=progress)
    print("\n")

    header = (
        f"{'Model':<26}{'Composite':>10}{'PA F1':>8}{'Seg':>7}"
        f"{'FA/day':>8}{'RMSE':>10}{'MASE':>8}{'Fit s':>8}"
    )
    print(header)
    print("-" * len(header))
    for m in result["models"]:
        d = m["detection"]
        f = m["forecast"] or {}

        def s(v, digits=3):
            return "-" if v is None else (f"{v:.{digits}f}" if isinstance(v, float) else str(v))

        marker = " *" if m["name"] == result["winner"] else "  "
        print(
            f"{m['display_name'][:24]+marker:<26}{s(m['composite_score']):>10}"
            f"{s(d['pa_f1']):>8}{s(d['segment_recall'],2):>7}"
            f"{s(d['false_alarms_per_day'],2):>8}{s(f.get('rmse')):>10}"
            f"{s(f.get('mase')):>8}{s(m['fit_seconds'],2):>8}"
        )

    print(f"\n{result['narrative']['headline']}")
    for point in result["narrative"]["points"]:
        print(f"  - {point}")

    print(f"\nIncidents raised: {len(result['incidents'])}")
    for inc in result["incidents"][:6]:
        confirmed = (
            "confirmed" if inc["confirmed_by_ground_truth"] else "unconfirmed"
        )
        print(
            f"  [{inc['severity']:>8}] {inc['start'][:16].replace('T',' ')}  "
            f"{inc['headline']}  ({confirmed})"
        )

    out = ROOT / "artifacts" / f"result_{dataset_id}.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFull result written to {out}")
    return 0


def cli_benchmark(freq: str, test_size: float) -> int:
    from sentinel.benchmark import run_benchmark
    from sentinel.config import RunConfig
    from sentinel.benchmark import labelled_dataset_ids

    ids = labelled_dataset_ids()
    template = RunConfig(dataset_id=ids[0], freq=freq, test_size=test_size)

    def progress(stage: str, pct: int) -> None:
        bar = "#" * (pct // 3)
        sys.stdout.write(f"\r  [{bar:<34}] {pct:3d}%  {stage[:38]:<38}")
        sys.stdout.flush()

    print(f"Benchmarking {len(ids)} labelled services\n")
    result = run_benchmark(base_config=template, progress=progress)
    print("\n")

    header = (
        f"{'Model':<26}{'Macro':>9}{'MacroPA':>9}{'PoolF1':>9}"
        f"{'Windows':>10}{'FA/day':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in result["summary"]:
        print(
            f"{row['display_name'][:25]:<26}"
            f"{(row['macro_composite'] or 0):>9.3f}"
            f"{(row['macro_pa_f1'] or 0):>9.3f}"
            f"{row['pooled_f1']:>9.3f}"
            f"{row['segments_detected']:>5}/{row['segments_total']:<4}"
            f"{(row['macro_false_alarms_per_day'] or 0):>8.2f}"
        )

    print()
    for note in result["notes"]:
        print(f"  - {note}")

    out = ROOT / "artifacts" / "benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"\nFull benchmark written to {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="SENTINEL-TS :: hybrid time-series anomaly intelligence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="auto-reload on edit")
    parser.add_argument("--check", action="store_true", help="verify setup and exit")
    parser.add_argument("--cli", metavar="DATASET_ID", help="run one analysis in the terminal")
    parser.add_argument("--benchmark", action="store_true", help="run the fleet benchmark")
    parser.add_argument("--freq", default="15min")
    parser.add_argument("--test-size", type=float, default=0.30)
    args = parser.parse_args()

    print(BANNER)

    if args.check:
        return 0 if check_environment() else 1

    if not check_environment():
        print("Environment check failed. Install the missing packages and retry:")
        print("    pip install -r requirements.txt\n")
        return 1

    if args.cli:
        return cli_run(args.cli, args.freq, args.test_size)
    if args.benchmark:
        return cli_benchmark(args.freq, args.test_size)

    import uvicorn

    url = f"http://{args.host}:{args.port}"
    print(f"Console  {url}")
    print(f"API docs {url}/docs")
    print("Press Ctrl+C to stop.\n")

    uvicorn.run(
        "sentinel.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
