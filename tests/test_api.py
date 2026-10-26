"""Integration tests for the HTTP layer and the end-to-end pipeline."""

from __future__ import annotations

import io
import time

import pytest
from fastapi.testclient import TestClient

from sentinel.api import create_app
from sentinel.config import RunConfig
from sentinel.datasets import list_datasets
from sentinel.incidents import alert_digest, build_incidents, capacity_outlook
from sentinel.pipeline import run_experiment


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app())


@pytest.fixture(scope="module")
def dataset_id():
    datasets = list_datasets()
    assert datasets, "no datasets registered; check data/raw"
    return datasets[0]["id"]


def wait_for(client, job_id, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("complete", "failed"):
            return st
        time.sleep(0.4)
    raise AssertionError("job did not finish within the timeout")


# ------------------------------------------------------------------ metadata


def test_health_reports_capabilities(client):
    body = client.get("/api/health").json()
    assert body["status"] == "ok"
    assert body["n_models"] >= 6
    assert body["n_datasets"] >= 1
    assert "15min" in body["supported_frequencies"]


def test_catalogue_entries_are_complete(client):
    body = client.get("/api/datasets").json()
    assert body["datasets"]
    for d in body["datasets"]:
        for key in ("id", "title", "n_obs", "start", "end", "unit", "spark"):
            assert key in d, f"missing {key}"
        assert d["n_obs"] > 0


def test_model_registry_exposes_rationales(client):
    body = client.get("/api/models").json()
    assert len(body["models"]) >= 6
    for m in body["models"]:
        assert m["rationale"], f"{m['name']} has no stated rationale"
        assert m["description"]


def test_preview_returns_plottable_series(client, dataset_id):
    body = client.get(f"/api/datasets/{dataset_id}/preview").json()
    assert len(body["timestamps"]) == len(body["values"])
    assert len(body["values"]) > 100


def test_preview_rejects_unknown_dataset(client):
    assert client.get("/api/datasets/not-a-dataset/preview").status_code == 404


def test_console_is_served(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "SENTINEL" in res.text


# ---------------------------------------------------------------- validation


def test_run_rejects_unknown_dataset(client):
    res = client.post("/api/run", json={"dataset_id": "nope"})
    assert res.status_code == 404


def test_run_rejects_unknown_model(client, dataset_id):
    res = client.post(
        "/api/run", json={"dataset_id": dataset_id, "models": ["telepathy"]}
    )
    assert res.status_code == 422


def test_run_rejects_bad_frequency(client, dataset_id):
    res = client.post("/api/run", json={"dataset_id": dataset_id, "freq": "3fortnights"})
    assert res.status_code == 422


def test_run_rejects_out_of_range_test_size(client, dataset_id):
    res = client.post("/api/run", json={"dataset_id": dataset_id, "test_size": 0.99})
    assert res.status_code == 422


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/deadbeef").status_code == 404


def test_result_before_completion_is_409(client, dataset_id):
    job = client.post(
        "/api/run",
        json={"dataset_id": dataset_id, "models": ["seasonal_naive", "rolling_zscore"]},
    ).json()
    res = client.get(f"/api/jobs/{job['id']}/result")
    assert res.status_code in (409, 200)  # may already be done on a fast machine
    wait_for(client, job["id"])


# ------------------------------------------------------------------- upload


def test_upload_accepts_a_valid_csv_and_registers_it(client):
    csv = "timestamp,value\n"
    csv += "".join(
        f"2024-03-01 {h:02d}:{m:02d}:00,{50 + h}\n" for h in range(24) for m in (0, 30)
    )
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("unit_test_series.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["dataset_id"].startswith("upload::")
    assert any(d["id"] == body["dataset_id"] for d in body["datasets"])


def test_upload_rejects_a_non_csv_extension(client):
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("notes.pdf", io.BytesIO(b"%PDF-1.4"), "application/pdf")},
    )
    assert res.status_code == 422


def test_upload_rejects_unparseable_content(client):
    res = client.post(
        "/api/datasets/upload",
        files={"file": ("junk.csv", io.BytesIO(b"this,is\nnot,timeseries\n"), "text/csv")},
    )
    assert res.status_code == 422


# ------------------------------------------------------------ end-to-end run


@pytest.fixture(scope="module")
def result(dataset_id):
    """One real pipeline run, shared by the assertions below."""
    config = RunConfig(dataset_id=dataset_id, freq="30min", test_size=0.3)
    return run_experiment(config)


def test_pipeline_bundle_has_every_section(result):
    for key in (
        "config", "dataset", "preprocessing", "eda", "split", "models",
        "model_series", "series", "winner", "incidents", "digest",
        "capacity_outlook", "narrative",
    ):
        assert key in result, f"missing section: {key}"


def test_every_requested_model_produced_a_row(result):
    names = {m["name"] for m in result["models"]}
    assert names == set(result["config"]["models"])
    assert all(m.get("error") is None for m in result["models"])


def test_models_are_ranked_contiguously_from_one(result):
    ranks = [m["rank"] for m in result["models"]]
    assert ranks == list(range(1, len(ranks) + 1))


def test_winner_is_the_top_ranked_model(result):
    assert result["winner"] == result["models"][0]["name"]


def test_split_is_chronological(result):
    assert result["split"]["train_end"] <= result["split"]["test_start"]
    assert result["split"]["n_train"] > result["split"]["n_test"]


def test_model_series_align_with_the_test_window(result):
    n = result["split"]["n_test"]
    assert len(result["series"]["test"]["values"]) == n
    for name, s in result["model_series"].items():
        assert len(s["scores"]) == n, name
        assert len(s["flags"]) == n, name
        if s["predictions"] is not None:
            assert len(s["predictions"]) == n, name


def test_eda_reports_a_reconciled_stationarity_verdict(result):
    verdict = result["eda"]["stationarity"]["verdict"]
    assert verdict["label"] in (
        "stationary", "non-stationary", "trend-stationary",
        "difference-stationary", "inconclusive",
    )
    assert verdict["recommended_d"] in (0, 1)
    assert verdict["explanation"]


def test_eda_produces_insights(result):
    assert len(result["eda"]["insights"]) >= 3


def test_incidents_carry_actionable_guidance(result):
    for inc in result["incidents"]:
        assert inc["severity"] in ("critical", "high", "medium", "low")
        assert inc["recommended_action"]
        assert inc["headline"]
        assert inc["end"] >= inc["start"]
        assert inc["duration_buckets"] >= 1


def test_bundle_is_json_serialisable(result):
    import json

    json.dumps(result)


# --------------------------------------------------------- incident helpers


def test_build_incidents_returns_empty_when_nothing_is_flagged():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=10, freq="15min")
    out = build_incidents(
        idx, np.ones(10), np.zeros(10, dtype=bool), np.zeros(10)
    )
    assert out == []


def test_build_incidents_merges_contiguous_flags():
    import numpy as np
    import pandas as pd

    idx = pd.date_range("2024-01-01", periods=12, freq="15min")
    actual = np.array([1.0] * 12)
    actual[4:8] = 50.0
    flags = np.zeros(12, dtype=bool)
    flags[4:8] = True
    incidents = build_incidents(idx, actual, flags, np.where(flags, 9.0, 0.1))
    assert len(incidents) == 1
    assert incidents[0]["duration_buckets"] == 4


def test_alert_digest_counts_by_severity():
    digest = alert_digest(
        [{"severity": "critical", "kind": "spike"}, {"severity": "low", "kind": "dip"}]
    )
    assert digest["total"] == 2
    assert digest["actionable"] == 1
    assert digest["by_kind"]["spike"] == 1


def test_capacity_outlook_handles_a_model_with_no_forecast():
    import numpy as np
    import pandas as pd

    out = capacity_outlook(
        pd.Timestamp("2024-01-01"), pd.Timedelta("15min"), None, "cpu", np.ones(10)
    )
    assert out["available"] is False
    assert out["message"]


def test_capacity_outlook_flags_a_ceiling_breach():
    import numpy as np
    import pandas as pd

    forecast = {
        "values": [50.0, 70.0, 95.0],
        "lower": [40.0, 60.0, 85.0],
        "upper": [60.0, 80.0, 105.0],
    }
    out = capacity_outlook(
        pd.Timestamp("2024-01-01"), pd.Timedelta("15min"), forecast, "cpu", np.ones(10)
    )
    assert out["breach_expected"] is True
    assert out["breach_index"] is not None


# ---------------------------------------------------------------- job export


def test_completed_job_exports_json(client, dataset_id):
    job = client.post(
        "/api/run",
        json={
            "dataset_id": dataset_id,
            "freq": "1h",
            "models": ["seasonal_naive", "rolling_zscore"],
        },
    ).json()
    st = wait_for(client, job["id"])
    assert st["status"] == "complete"

    res = client.get(f"/api/jobs/{job['id']}/export")
    assert res.status_code == 200
    assert "attachment" in res.headers["content-disposition"]
    assert res.json()["winner"]
