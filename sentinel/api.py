"""
SENTINEL-TS :: HTTP API and console host.

The API is deliberately thin. It validates input, delegates to the analysis
package and serialises the result; it contains no modelling logic of its own,
which is what allows the science to be tested without a server running.

Endpoints
---------
``GET  /api/health``               Liveness and capability probe.
``GET  /api/datasets``             Catalogue, including user uploads.
``GET  /api/models``               Model registry with rationales.
``POST /api/datasets/upload``      Accept a user CSV and register it.
``POST /api/run``                  Queue an analysis; returns a job id.
``POST /api/benchmark``            Queue the cross-dataset benchmark.
``GET  /api/jobs``                 All known jobs.
``GET  /api/jobs/{id}``            Status and progress.
``GET  /api/jobs/{id}/result``     Full result bundle once complete.
``GET  /api/jobs/{id}/export``     Result bundle as a downloadable file.
"""

from __future__ import annotations

import json
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .benchmark import labelled_dataset_ids, run_benchmark
from .config import (
    DEFAULT_CONTAMINATION,
    DEFAULT_FORECAST_HORIZON,
    DEFAULT_FOURIER_K,
    DEFAULT_FREQ,
    DEFAULT_ROLLING_WINDOW,
    DEFAULT_TEST_SIZE,
    DEFAULT_THRESHOLD_SIGMA,
    FRONTEND_DIR,
    PERIODS_PER_DAY,
    RunConfig,
)
from .datasets import list_datasets, save_upload
from .datasets import preview as dataset_preview_payload
from .jobs import JobManager
from .models import REGISTRY, TORCH_AVAILABLE, available_models
from .pipeline import run_experiment

MAX_UPLOAD_BYTES = 8 * 1024 * 1024


class RunRequest(BaseModel):
    """Validated run configuration from the console."""

    dataset_id: str = Field(..., min_length=1, max_length=200)
    freq: str = DEFAULT_FREQ
    test_size: float = Field(DEFAULT_TEST_SIZE, gt=0.05, lt=0.9)
    threshold_sigma: float = Field(DEFAULT_THRESHOLD_SIGMA, gt=0.5, le=12.0)
    fourier_k: int = Field(DEFAULT_FOURIER_K, ge=1, le=10)
    rolling_window: int = Field(DEFAULT_ROLLING_WINDOW, ge=8, le=1000)
    contamination: float = Field(DEFAULT_CONTAMINATION, gt=0.0005, le=0.25)
    horizon: int = Field(DEFAULT_FORECAST_HORIZON, ge=4, le=1000)
    models: Optional[List[str]] = None


class BenchmarkRequest(BaseModel):
    freq: str = DEFAULT_FREQ
    test_size: float = Field(DEFAULT_TEST_SIZE, gt=0.05, lt=0.9)
    threshold_sigma: float = Field(DEFAULT_THRESHOLD_SIGMA, gt=0.5, le=12.0)
    dataset_ids: Optional[List[str]] = None
    models: Optional[List[str]] = None


def _to_config(req: RunRequest) -> RunConfig:
    if req.freq not in PERIODS_PER_DAY:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported frequency '{req.freq}'. "
                f"Choose one of: {', '.join(PERIODS_PER_DAY)}."
            ),
        )

    models = req.models or list(RunConfig(dataset_id=req.dataset_id).models)
    unknown = [m for m in models if m not in REGISTRY]
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown model(s): {', '.join(unknown)}.",
        )

    return RunConfig(
        dataset_id=req.dataset_id,
        freq=req.freq,
        test_size=req.test_size,
        fourier_k=req.fourier_k,
        threshold_sigma=req.threshold_sigma,
        rolling_window=req.rolling_window,
        contamination=req.contamination,
        horizon=req.horizon,
        models=models,
    )


def create_app() -> FastAPI:
    app = FastAPI(
        title="SENTINEL-TS",
        version=__version__,
        description=(
            "Hybrid time-series anomaly intelligence for cloud infrastructure "
            "telemetry."
        ),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    jobs = JobManager(max_workers=2)
    app.state.jobs = jobs

    # ---------------- metadata -------------------------------------------

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "version": __version__,
            "torch_available": TORCH_AVAILABLE,
            "n_datasets": len(list_datasets()),
            "n_models": len(REGISTRY),
            "supported_frequencies": list(PERIODS_PER_DAY.keys()),
        }

    @app.get("/api/datasets")
    def datasets():
        return {"datasets": list_datasets()}

    @app.get("/api/datasets/{dataset_id}/preview")
    def dataset_preview(dataset_id: str):
        try:
            return dataset_preview_payload(dataset_id)
        except (KeyError, FileNotFoundError):
            raise HTTPException(
                status_code=404, detail=f"Dataset '{dataset_id}' not found."
            )

    @app.get("/api/models")
    def models():
        return {"models": available_models(), "torch_available": TORCH_AVAILABLE}

    @app.get("/api/defaults")
    def defaults():
        return {
            "freq": DEFAULT_FREQ,
            "test_size": DEFAULT_TEST_SIZE,
            "threshold_sigma": DEFAULT_THRESHOLD_SIGMA,
            "fourier_k": DEFAULT_FOURIER_K,
            "rolling_window": DEFAULT_ROLLING_WINDOW,
            "contamination": DEFAULT_CONTAMINATION,
            "horizon": DEFAULT_FORECAST_HORIZON,
            "frequencies": list(PERIODS_PER_DAY.keys()),
            "models": list(REGISTRY.keys()),
        }

    # ---------------- upload ----------------------------------------------

    @app.post("/api/datasets/upload")
    async def upload(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith((".csv", ".txt")):
            raise HTTPException(
                status_code=422, detail="Please upload a .csv file."
            )
        content = await file.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024*1024)} MB limit.",
            )
        try:
            dataset_id = save_upload(file.filename, content)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Could not parse that file. A timestamp column and a "
                    f"numeric value column are required. ({exc})"
                ),
            )
        return {"dataset_id": dataset_id, "datasets": list_datasets()}

    # ---------------- runs -------------------------------------------------

    @app.post("/api/run")
    def run(req: RunRequest):
        config = _to_config(req)
        available = {d["id"] for d in list_datasets()}
        if config.dataset_id not in available:
            raise HTTPException(
                status_code=404, detail=f"Dataset '{config.dataset_id}' not found."
            )

        job = jobs.submit(
            kind="run",
            label=config.dataset_id,
            fn=lambda progress, cfg=config: run_experiment(cfg, progress=progress),
        )
        return JSONResponse(status_code=202, content=job.public())

    @app.post("/api/benchmark")
    def benchmark(req: BenchmarkRequest):
        ids = req.dataset_ids or labelled_dataset_ids()
        if not ids:
            raise HTTPException(
                status_code=400,
                detail="No labelled datasets are available to benchmark.",
            )
        template = _to_config(
            RunRequest(
                dataset_id=ids[0],
                freq=req.freq,
                test_size=req.test_size,
                threshold_sigma=req.threshold_sigma,
                models=req.models,
            )
        )
        job = jobs.submit(
            kind="benchmark",
            label=f"{len(ids)} datasets",
            fn=lambda progress, i=ids, t=template: run_benchmark(
                dataset_ids=i, base_config=t, progress=progress
            ),
        )
        return JSONResponse(status_code=202, content=job.public())

    # ---------------- jobs -------------------------------------------------

    @app.get("/api/jobs")
    def all_jobs():
        return {"jobs": jobs.list()}

    @app.get("/api/jobs/{job_id}")
    def job_status(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        return job.public()

    @app.get("/api/jobs/{job_id}/result")
    def job_result(job_id: str):
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Unknown job id.")
        if job.status == "failed":
            raise HTTPException(status_code=500, detail=job.error or "Run failed.")
        if job.status != "complete":
            raise HTTPException(
                status_code=409,
                detail=f"Job is {job.status} ({job.progress}%). Poll /api/jobs/{job_id}.",
            )
        return job.public(include_result=True)

    @app.get("/api/jobs/{job_id}/export")
    def job_export(job_id: str):
        job = jobs.get(job_id)
        if job is None or job.status != "complete":
            raise HTTPException(status_code=404, detail="No completed result for that id.")
        payload = json.dumps(job.result, indent=2)
        return Response(
            content=payload,
            media_type="application/json",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="sentinel_{job.kind}_{job.id}.json"'
                )
            },
        )

    # ---------------- console ---------------------------------------------

    index_file = FRONTEND_DIR / "index.html"

    @app.get("/")
    def console():
        if not index_file.exists():  # pragma: no cover
            return JSONResponse(
                {"detail": "Console assets not found; the API is still available."},
                status_code=404,
            )
        return FileResponse(index_file)

    if FRONTEND_DIR.exists():
        app.mount(
            "/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static"
        )

    return app


app = create_app()
