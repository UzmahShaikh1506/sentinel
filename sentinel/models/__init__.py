"""SENTINEL-TS :: Model registry."""

from __future__ import annotations

from typing import Dict, Type

from .base import BaseDetector, ModelOutput, normalise_scores
from .baselines import RollingZScore, SeasonalNaive
from .classical import HoltWinters, SarimaxFourier
from .hybrid import HybridDetector, HybridNeuralDetector, debounce
from .ml import IsolationForestDetector, ResidualLearner, TORCH_AVAILABLE

#: Every model the comparison harness can instantiate, in presentation order:
#: baselines first, then classical, then machine learning, then the hybrids.
REGISTRY: Dict[str, Type[BaseDetector]] = {
    "seasonal_naive": SeasonalNaive,
    "rolling_zscore": RollingZScore,
    "holt_winters": HoltWinters,
    "sarimax_fourier": SarimaxFourier,
    "isolation_forest": IsolationForestDetector,
    "hybrid": HybridDetector,
    "hybrid_neural": HybridNeuralDetector,
}

DEFAULT_MODELS = list(REGISTRY.keys())


def available_models() -> list:
    """Catalogue entries for the console's model picker."""
    out = []
    for name, cls in REGISTRY.items():
        out.append(
            {
                "name": name,
                "display_name": cls.display_name,
                "family": cls.family,
                "description": cls.description,
                "rationale": cls.rationale,
                "forecasts": cls.forecasts,
            }
        )
    return out


__all__ = [
    "BaseDetector",
    "ModelOutput",
    "REGISTRY",
    "DEFAULT_MODELS",
    "available_models",
    "normalise_scores",
    "debounce",
    "ResidualLearner",
    "TORCH_AVAILABLE",
]
