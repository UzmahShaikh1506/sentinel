"""Tests for the forecast and detection scorecards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sentinel.metrics import (
    composite_score,
    detection_metrics,
    forecast_metrics,
    point_adjust,
    rank_models,
)


# ------------------------------------------------------------------- forecast


def test_perfect_forecast_scores_zero_error():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    m = forecast_metrics(a, a.copy())
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mape"] == 0.0
    assert m["r2"] == 1.0


def test_forecast_metrics_are_arithmetically_correct():
    a = np.array([10.0, 20.0, 30.0])
    p = np.array([12.0, 18.0, 33.0])
    m = forecast_metrics(a, p)
    # Reported values are rounded to four decimals for display.
    assert m["mae"] == pytest.approx((2 + 2 + 3) / 3, abs=1e-4)
    assert m["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3), abs=1e-4)


def test_mape_skips_zero_actuals_rather_than_exploding():
    a = np.array([0.0, 10.0, 0.0, 20.0])
    p = np.array([1.0, 11.0, 2.0, 18.0])
    m = forecast_metrics(a, p)
    assert np.isfinite(m["mape"])
    # Only the two non-zero actuals contribute: 10% and 10%.
    assert m["mape"] == pytest.approx(10.0)


def test_mase_is_scaled_by_the_seasonal_naive_benchmark():
    rng = np.random.default_rng(0)
    insample = np.tile(np.arange(10, dtype="float64"), 10) + rng.normal(0, 0.01, 100)
    actual = np.arange(10, dtype="float64")
    m = forecast_metrics(actual, actual.copy(), insample=insample, seasonal_period=10)
    assert m["mase"] == pytest.approx(0.0, abs=1e-6)


def test_forecast_metrics_ignore_non_finite_pairs():
    a = np.array([1.0, np.nan, 3.0])
    p = np.array([1.0, 2.0, np.inf])
    m = forecast_metrics(a, p)
    assert m["n"] == 1


def test_forecast_metrics_handle_empty_input():
    m = forecast_metrics(np.array([]), np.array([]))
    assert m["mae"] is None


# ------------------------------------------------------------------ detection


def test_point_adjust_credits_a_whole_hit_segment():
    truth = np.array([False, True, True, True, False])
    pred = np.array([False, False, True, False, False])
    assert point_adjust(pred, truth).tolist() == [False, True, True, True, False]


def test_point_adjust_leaves_a_missed_segment_alone():
    truth = np.array([False, True, True, False])
    pred = np.array([False, False, False, False])
    assert point_adjust(pred, truth).tolist() == [False, False, False, False]


def test_point_adjust_never_invents_positives_outside_truth():
    truth = np.array([False, False, True, False])
    pred = np.array([True, False, False, False])
    out = point_adjust(pred, truth)
    assert out[0]           # the original false positive is preserved
    assert not out[2]       # the missed segment is not credited


def test_point_adjust_handles_a_segment_touching_the_end():
    truth = np.array([False, True, True])
    pred = np.array([False, False, True])
    assert point_adjust(pred, truth).tolist() == [False, True, True]


def test_detection_metrics_confusion_is_correct():
    truth = np.array([False, True, True, False, False])
    pred = np.array([False, True, False, True, False])
    idx = pd.date_range("2024-01-01", periods=5, freq="15min")
    m = detection_metrics(pred, truth, idx)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 2)
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1"] == pytest.approx(0.5)


def test_detection_metrics_measure_latency_in_minutes():
    truth = np.array([False, False, True, True, True, False])
    pred = np.array([False, False, False, True, False, False])
    idx = pd.date_range("2024-01-01", periods=6, freq="15min")
    m = detection_metrics(pred, truth, idx)
    assert m["mean_detection_latency_points"] == pytest.approx(1.0)
    assert m["mean_detection_latency_minutes"] == pytest.approx(15.0)
    assert m["n_segments_detected"] == 1
    assert m["segment_recall"] == 1.0


def test_detection_metrics_without_labels_return_none_not_zero():
    """An unlabelled series must not be reported as a perfect or failed score."""
    pred = np.array([False, True, False])
    idx = pd.date_range("2024-01-01", periods=3, freq="15min")
    m = detection_metrics(pred, None, idx)
    assert m["labelled"] is False
    assert m["f1"] is None and m["precision"] is None
    assert m["n_flagged"] == 1


def test_false_alarm_episodes_exclude_overlapping_alerts():
    truth = np.array([False, True, True, False, False, False])
    pred = np.array([False, True, False, False, True, False])
    idx = pd.date_range("2024-01-01", periods=6, freq="1h")
    m = detection_metrics(pred, truth, idx)
    assert m["n_alert_episodes"] == 2
    # One episode overlaps truth, one does not -> exactly one false alarm.
    assert m["false_alarms_per_day"] > 0


# -------------------------------------------------------------------- ranking


def test_composite_score_is_none_without_labels():
    assert composite_score({"labelled": False}) is None


def test_composite_score_penalises_alert_burden():
    base = {
        "labelled": True,
        "pa_f1": 1.0,
        "segment_recall": 1.0,
        "f1": 1.0,
        "mean_detection_latency_points": 0,
        "false_alarms_per_day": 0.0,
    }
    noisy = dict(base, false_alarms_per_day=20.0)
    assert composite_score(base) > composite_score(noisy)


def test_composite_score_penalises_latency():
    base = {
        "labelled": True,
        "pa_f1": 1.0,
        "segment_recall": 1.0,
        "f1": 1.0,
        "mean_detection_latency_points": 0,
        "false_alarms_per_day": 0.0,
    }
    slow = dict(base, mean_detection_latency_points=48)
    assert composite_score(base) > composite_score(slow)


def test_rank_models_orders_best_first_and_assigns_ranks():
    rows = [
        {"name": "a", "composite_score": 0.5, "detection": {}, "forecast": {}},
        {"name": "b", "composite_score": 0.9, "detection": {}, "forecast": {}},
        {"name": "c", "composite_score": 0.7, "detection": {}, "forecast": {}},
    ]
    ranked = rank_models(rows)
    assert [r["name"] for r in ranked] == ["b", "c", "a"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]


def test_rank_models_places_unscored_models_last():
    rows = [
        {"name": "unlabelled", "composite_score": None,
         "detection": {"flag_rate": 0.1}, "forecast": {"rmse": 1.0}},
        {"name": "scored", "composite_score": 0.4, "detection": {}, "forecast": {}},
    ]
    ranked = rank_models(rows)
    assert ranked[0]["name"] == "scored"
