"""
SENTINEL-TS
===========

Hybrid time-series anomaly intelligence for cloud infrastructure telemetry.

A forecasting and anomaly-detection system that compares six approaches on real
Amazon CloudWatch data and serves the winner through an operations console:
seasonal naive and rolling z-score baselines, Holt-Winters ETS, SARIMAX with
Fourier seasonality, Isolation Forest, and a two-stage hybrid that corrects a
linear forecast with a nonlinear residual learner before fusing two
complementary detectors.

Public surface
--------------
``run_experiment``
    End-to-end analysis returning a JSON-ready result bundle.
``RunConfig``
    Everything that parameterises a run.
``create_app``
    FastAPI application serving the REST API and the console.
"""

__version__ = "1.0.0"
__all__ = ["RunConfig", "run_experiment", "create_app", "__version__"]

from .config import RunConfig  # noqa: E402
from .pipeline import run_experiment  # noqa: E402


def create_app():  # pragma: no cover - thin re-export
    """Build the FastAPI application (imported lazily to keep CLI use light)."""
    from .api import create_app as _create_app

    return _create_app()
