"""Tests for cleaning, resampling and the robust statistics built on them."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.preprocessing import (
    MAD_TO_SIGMA,
    preprocess,
    robust_scale,
    rolling_robust_zscore,
    seasonal_profile,
    train_test_split,
    windows_to_point_labels,
)


def make_series(n=400, freq="15min", start="2024-01-01"):
    idx = pd.date_range(start, periods=n, freq=freq)
    t = np.arange(n)
    values = 50 + 10 * np.sin(2 * np.pi * t / 96) + np.random.default_rng(0).normal(0, 1, n)
    return pd.Series(values, index=idx, name="value")


# ---------------------------------------------------------------- robust stats


def test_robust_scale_matches_mad_definition():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = np.median(np.abs(values - np.median(values))) * MAD_TO_SIGMA
    assert robust_scale(values) == pytest.approx(expected)


def test_robust_scale_resists_extreme_contamination():
    clean = np.random.default_rng(1).normal(0, 1, 500)
    contaminated = np.concatenate([clean, [1e6, -1e6]])
    # A standard deviation would explode; the MAD estimate should barely move.
    assert robust_scale(contaminated) == pytest.approx(robust_scale(clean), rel=0.1)
    assert np.std(contaminated) > 100 * robust_scale(contaminated)


def test_robust_scale_falls_back_on_constant_input():
    """A long constant run collapses the MAD to zero; the result must stay usable."""
    assert robust_scale(np.zeros(100)) > 0


def test_rolling_zscore_is_causal():
    """A spike must not raise the score of points before it."""
    s = make_series(300)
    spiked = s.copy()
    spiked.iloc[200] = 500.0

    z_before = rolling_robust_zscore(s, 48).iloc[:190]
    z_after = rolling_robust_zscore(spiked, 48).iloc[:190]
    assert np.allclose(z_before.to_numpy(), z_after.to_numpy(), equal_nan=True)


def test_rolling_zscore_flags_the_spike():
    s = make_series(300)
    s.iloc[200] = 500.0
    z = rolling_robust_zscore(s, 48)
    assert abs(z.iloc[200]) > 10


# ----------------------------------------------------------------- preprocess


def test_preprocess_regularises_grid_and_reports():
    s = make_series(400, freq="5min")
    clean, flags, report = preprocess(s, freq="15min")

    assert isinstance(clean.index, pd.DatetimeIndex)
    assert clean.index.freq is not None or len(set(np.diff(clean.index.values))) == 1
    assert not clean.isna().any()
    assert report.n_input == 400
    assert report.n_output == len(clean)
    assert len(flags) == len(clean)


def test_preprocess_collapses_duplicate_timestamps():
    idx = pd.DatetimeIndex(["2024-01-01 00:00", "2024-01-01 00:00", "2024-01-01 00:15"])
    s = pd.Series([10.0, 20.0, 30.0], index=idx)
    clean, _, report = preprocess(s, freq="15min")
    assert report.n_duplicate_timestamps == 1
    assert clean.iloc[0] == pytest.approx(15.0)  # mean of 10 and 20


def test_preprocess_masks_non_finite_values():
    s = make_series(200)
    s.iloc[10] = np.inf
    s.iloc[20] = np.nan
    _, _, report = preprocess(s, freq="15min")
    assert report.n_non_finite >= 1


def test_short_gaps_interpolate_and_long_gaps_use_seasonal_profile():
    s = make_series(600, freq="15min")
    s.iloc[100:103] = np.nan      # short gap -> interpolation
    s.iloc[300:340] = np.nan      # long gap  -> seasonal reconstruction
    clean, _, report = preprocess(s, freq="15min")

    assert not clean.isna().any()
    assert report.n_filled_interpolated >= 3
    assert report.n_filled_seasonal >= 40
    # A seasonally reconstructed stretch should not be a straight line.
    reconstructed = clean.iloc[300:340].to_numpy()
    assert np.std(np.diff(reconstructed)) > 1e-6


def test_outliers_are_flagged_not_removed():
    s = make_series(400)
    s.iloc[150] = 9999.0
    clean, flags, report = preprocess(s, freq="15min")
    assert report.n_outliers_flagged >= 1
    assert clean.max() > 1000          # the extreme value survived
    assert flags.sum() == report.n_outliers_flagged


def test_preprocess_rejects_empty_and_non_datetime_input():
    with pytest.raises(ValueError):
        preprocess(pd.Series(dtype="float64"), freq="15min")
    with pytest.raises(TypeError):
        preprocess(pd.Series([1.0, 2.0], index=[0, 1]), freq="15min")


# ----------------------------------------------------------------- split/labels


def test_train_test_split_is_chronological_and_complete():
    s = make_series(500)
    train, test = train_test_split(s, 0.3)
    assert len(train) + len(test) == len(s)
    assert train.index[-1] < test.index[0]
    assert len(test) == pytest.approx(150, abs=1)


def test_train_test_split_validates_bounds():
    s = make_series(100)
    for bad in (0.0, 1.0, -0.2, 1.5):
        with pytest.raises(ValueError):
            train_test_split(s, bad)


def test_windows_to_point_labels_marks_inclusive_range():
    idx = pd.date_range("2024-01-01", periods=10, freq="1h")
    windows = [(pd.Timestamp("2024-01-01 02:00"), pd.Timestamp("2024-01-01 04:00"))]
    labels = windows_to_point_labels(idx, windows)
    assert labels.tolist() == [False, False, True, True, True, False, False, False, False, False]


def test_seasonal_profile_recovers_a_known_cycle():
    n, period = 480, 96
    t = np.arange(n)
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    s = pd.Series(np.sin(2 * np.pi * t / period), index=idx)
    profile = seasonal_profile(s, period)
    assert len(profile) == period
    assert profile.idxmax() == pytest.approx(period // 4, abs=2)
