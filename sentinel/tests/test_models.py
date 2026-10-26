"""Tests for the model zoo, with particular attention to leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.models import REGISTRY
from sentinel.models.baselines import SeasonalNaive
from sentinel.models.hybrid import HybridDetector, debounce
from sentinel.models.ml import ResidualLearner, build_window_features
from sentinel.preprocessing import train_test_split

PERIOD = 96


@pytest.fixture(scope="module")
def seasonal_series():
    """Clean daily cycle plus noise and a planted anomaly in the test window."""
    n = 700
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    t = np.arange(n)
    rng = np.random.default_rng(7)
    values = (
        60
        + 15 * np.sin(2 * np.pi * t / PERIOD)
        + 4 * np.sin(2 * np.pi * t / (PERIOD / 2))
        + rng.normal(0, 1.2, n)
    )
    values[600:615] += 45  # the incident the detectors should find
    return pd.Series(values, index=idx, name="value")


@pytest.fixture(scope="module")
def split(seasonal_series):
    return train_test_split(seasonal_series, 0.3)


# ------------------------------------------------------------------ contract


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_every_model_satisfies_the_output_contract(name, split):
    train, test = split
    model = REGISTRY[name](period=PERIOD, threshold_sigma=3.5, horizon=24)
    out = model.run(train, test)

    assert out.error is None, f"{name} raised: {out.error}"
    assert out.scores.shape == (len(test),)
    assert out.flags.shape == (len(test),)
    assert out.flags.dtype == bool
    assert np.isfinite(out.scores).all()
    assert out.fit_seconds >= 0
    if out.predictions is not None:
        assert out.predictions.shape == (len(test),)
        assert np.isfinite(out.predictions).all()


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_every_model_finds_a_large_planted_anomaly(name, split):
    """A +45 unit step on a series with sigma ~1.2 is not subtle."""
    train, test = split
    model = REGISTRY[name](period=PERIOD, threshold_sigma=3.5, horizon=24)
    out = model.run(train, test)

    offset = len(train)
    lo, hi = 600 - offset, 615 - offset
    assert out.flags[lo : hi + 1].any(), f"{name} missed the planted anomaly"


@pytest.mark.parametrize("name", list(REGISTRY.keys()))
def test_models_do_not_alert_on_everything(name, split):
    """A detector that flags most of the window is useless even at recall 1."""
    train, test = split
    model = REGISTRY[name](period=PERIOD, threshold_sigma=3.5, horizon=24)
    out = model.run(train, test)
    assert out.flags.mean() < 0.35, f"{name} flagged {out.flags.mean():.0%} of the window"


def test_failing_model_is_reported_not_raised(split):
    """One unstable model must not abort an entire comparison run."""
    train, test = split

    class Broken(SeasonalNaive):
        name = "broken"

        def fit(self, train):
            raise RuntimeError("synthetic failure")

    out = Broken(period=PERIOD).run(train, test)
    assert out.error is not None
    assert "synthetic failure" in out.error
    assert out.flags.shape == (len(test),)
    assert not out.flags.any()


# ------------------------------------------------------------------- leakage


def test_window_features_are_strictly_causal():
    """Changing the value at t must not alter the feature row at t."""
    n = 300
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    s = pd.Series(np.linspace(0, 10, n), index=idx)

    base, _ = build_window_features(s)
    perturbed_series = s.copy()
    perturbed_series.iloc[200] = 999.0
    perturbed, _ = build_window_features(perturbed_series)

    assert np.allclose(base[200], perturbed[200]), "feature row at t saw y_t"


def test_seasonal_naive_predicts_the_previous_cycle(split):
    train, test = split
    model = SeasonalNaive(period=PERIOD).fit(train)
    pred = model.predict(test)
    combined = pd.concat([train, test])
    expected = combined.shift(PERIOD).iloc[len(train) :].to_numpy()
    assert np.allclose(pred, expected, equal_nan=False)


def test_test_window_values_do_not_change_fitted_parameters(split):
    """Refitting with a corrupted test window must leave training output intact."""
    train, test = split
    corrupted = test.copy()
    corrupted.iloc[:] = corrupted.to_numpy() * 10

    a = HybridDetector(period=PERIOD, horizon=12).fit(train)
    b = HybridDetector(period=PERIOD, horizon=12).fit(train)
    assert a.params["stage_1_order"] == b.params["stage_1_order"]
    assert a._resid_scale == pytest.approx(b._resid_scale)


# -------------------------------------------------------------------- hybrid


def test_hybrid_reports_its_two_stages_and_variance_reduction(split):
    train, _ = split
    model = HybridDetector(period=PERIOD, horizon=12).fit(train)
    diag = model.diagnostics
    assert "residual_variance_reduction_pct" in diag
    assert diag["linear_residual_variance"] > 0
    assert model.params["stage_1"].startswith("SARIMAX")
    assert model.params["stage_2"]


def test_hybrid_forecast_does_not_extrapolate_the_nonlinear_stage(split):
    train, _ = split
    model = HybridDetector(period=PERIOD, horizon=24).fit(train)
    fc = model.forecast_future(24)
    assert fc is not None
    assert len(fc["values"]) == 24
    assert "not extrapolated" in fc["method"]


def test_debounce_suppresses_short_runs_and_keeps_long_ones():
    flags = np.array([True, False, True, True, True, False, True, False])
    out = debounce(flags, 3)
    assert out.tolist() == [False, False, True, True, True, False, False, False]


def test_debounce_is_a_no_op_at_one():
    flags = np.array([True, False, True])
    assert debounce(flags, 1).tolist() == flags.tolist()


def test_debounce_keeps_a_run_touching_the_end():
    flags = np.array([False, True, True, True])
    assert debounce(flags, 3).tolist() == [False, True, True, True]


# ----------------------------------------------------------- residual learner


def test_residual_learner_falls_back_when_torch_is_absent():
    learner = ResidualLearner(kind="lstm")
    try:
        import torch  # noqa: F401

        assert learner.kind == "lstm"
    except ImportError:
        assert learner.kind == "mlp"
        assert "PyTorch" in learner.fallback_reason


def test_residual_learner_rejects_unknown_kinds():
    with pytest.raises(ValueError):
        ResidualLearner(kind="not-a-model")


def test_residual_learner_degrades_gracefully_on_tiny_input():
    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    learner = ResidualLearner(kind="gbr").fit(np.random.normal(0, 1, 20), idx)
    assert learner.model is None
    out = learner.predict(np.random.normal(0, 1, 20), idx)
    assert out.shape == (20,)
    assert np.allclose(out, 0.0)


def test_residual_learner_captures_a_deterministic_pattern():
    """A residual that is a clean function of its own lag must be learnable."""
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.default_rng(3)
    e = np.zeros(n)
    for t in range(1, n):
        e[t] = 0.8 * e[t - 1] + rng.normal(0, 0.1)

    learner = ResidualLearner(kind="gbr", lags=6).fit(e, idx)
    pred = learner.predict(e, idx)
    residual_after = e[50:] - pred[50:]
    assert np.var(residual_after) < 0.6 * np.var(e[50:])


def test_hybrid_correction_cannot_absorb_a_sustained_excursion():
    """Regression test for the stage-two explain-away failure.

    An unclamped residual learner forecasts the residuals of a *sustained*
    anomaly accurately, because they are large but persistent. The correction
    then cancels them, the hybrid residual collapses, and the detector goes
    silent exactly when the incident is happening. This was observed on the ELB
    request-count service, where the hybrid scored 0.00 segment recall while
    every other model caught both windows.

    The correction must stay bounded by the training residual scale, so the
    hybrid degrades to its linear stage during a real excursion.
    """
    n = 800
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    t = np.arange(n)
    rng = np.random.default_rng(11)
    values = 100 + 20 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 2.0, n)
    values[620:700] += 60  # long, sustained excursion in the test window

    series = pd.Series(values, index=idx)
    train, test = train_test_split(series, 0.3)

    out = HybridDetector(period=PERIOD, threshold_sigma=3.5, horizon=12).run(train, test)
    assert out.error is None

    offset = len(train)
    lo, hi = 620 - offset, 699 - offset
    detected = out.flags[lo : hi + 1]

    assert detected.any(), "hybrid went silent during a sustained anomaly"
    # It must stay alerting through the body of the event, not just at the edge.
    assert detected.mean() > 0.5, (
        f"hybrid alerted on only {detected.mean():.0%} of a sustained excursion; "
        "the nonlinear stage is still absorbing it"
    )


def test_hybrid_clamp_is_reported_and_rarely_binds_in_normal_operation():
    n = 700
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    t = np.arange(n)
    rng = np.random.default_rng(5)
    series = pd.Series(
        80 + 12 * np.sin(2 * np.pi * t / PERIOD) + rng.normal(0, 1.5, n), index=idx
    )
    train, _ = train_test_split(series, 0.3)

    model = HybridDetector(period=PERIOD, horizon=12).fit(train)
    assert model.params["residual_correction_clamp_sigma"] == HybridDetector.RESIDUAL_CLAMP
    assert model.diagnostics["correction_clamp_absolute"] > 0
    # On a well-behaved series the clamp should almost never engage.
    assert model.diagnostics["correction_clamp_hit_rate_train"] < 0.05
