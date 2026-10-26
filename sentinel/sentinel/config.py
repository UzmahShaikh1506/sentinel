"""
SENTINEL-TS :: Global configuration.

Central place for paths, modelling defaults and the service-level thresholds
that drive the decision-support layer. Nothing here is imported for its side
effects; everything is a plain constant or a small dataclass so that the
configuration can be unit-tested and overridden from the API layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LABEL_DIR = DATA_DIR / "labels"
UPLOAD_DIR = DATA_DIR / "uploads"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

for _d in (UPLOAD_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LABEL_FILE = LABEL_DIR / "combined_windows.json"

# --------------------------------------------------------------------------
# Modelling defaults
# --------------------------------------------------------------------------

#: Native sampling interval of the NAB AWS CloudWatch corpus.
NATIVE_FREQ = "5min"

#: Default resampling target. 15-minute buckets keep roughly 1 300 observations
#: over the two-week horizon, which is long enough for a daily seasonal profile
#: (96 buckets/day) while keeping SARIMAX estimation inside interactive latency.
DEFAULT_FREQ = "15min"

#: Fraction of the series held out, chronologically, for evaluation.
DEFAULT_TEST_SIZE = 0.30

#: Number of Fourier harmonic pairs used to encode the daily cycle inside
#: SARIMAX. Long seasonal periods (m = 96) are intractable for a classical
#: seasonal ARIMA, so the seasonality is moved into exogenous regressors.
DEFAULT_FOURIER_K = 4

#: Seasonal period, expressed in buckets per day, per supported frequency.
PERIODS_PER_DAY: Dict[str, int] = {
    "5min": 288,
    "10min": 144,
    "15min": 96,
    "30min": 48,
    "1h": 24,
}

#: Robust-z threshold at which a residual is called anomalous.
DEFAULT_THRESHOLD_SIGMA = 3.5

#: Rolling window (in buckets) for the robust location/scale estimate.
DEFAULT_ROLLING_WINDOW = 96

#: Expected contamination passed to Isolation Forest.
DEFAULT_CONTAMINATION = 0.03

#: Horizon, in buckets, for the forward capacity forecast shown in the UI.
DEFAULT_FORECAST_HORIZON = 96

RANDOM_SEED = 42


@dataclass
class RunConfig:
    """Everything the pipeline needs for one end-to-end experiment."""

    dataset_id: str
    freq: str = DEFAULT_FREQ
    test_size: float = DEFAULT_TEST_SIZE
    fourier_k: int = DEFAULT_FOURIER_K
    threshold_sigma: float = DEFAULT_THRESHOLD_SIGMA
    rolling_window: int = DEFAULT_ROLLING_WINDOW
    contamination: float = DEFAULT_CONTAMINATION
    horizon: int = DEFAULT_FORECAST_HORIZON
    models: list = field(
        default_factory=lambda: [
            "seasonal_naive",
            "rolling_zscore",
            "holt_winters",
            "sarimax_fourier",
            "isolation_forest",
            "hybrid",
            "hybrid_neural",
        ]
    )
    use_deep_residual: bool = True

    @property
    def seasonal_period(self) -> int:
        return PERIODS_PER_DAY.get(self.freq, 96)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["seasonal_period"] = self.seasonal_period
        return d


# --------------------------------------------------------------------------
# Decision support: service-level thresholds per metric family
# --------------------------------------------------------------------------
#: Each entry maps a metric family to the operational ceiling above which an
#: SRE would be expected to act, plus the unit used in the UI.
SLO_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "cpu": {"ceiling": 85.0, "unit": "%", "direction": "upper"},
    "network": {"ceiling": None, "unit": "bytes", "direction": "upper"},
    "disk": {"ceiling": None, "unit": "bytes", "direction": "upper"},
    "requests": {"ceiling": None, "unit": "req/bucket", "direction": "both"},
    "generic": {"ceiling": None, "unit": "units", "direction": "both"},
}

#: Severity banding applied to the normalised anomaly score.
SEVERITY_BANDS = [
    (0.90, "critical"),
    (0.75, "high"),
    (0.55, "medium"),
    (0.00, "low"),
]
