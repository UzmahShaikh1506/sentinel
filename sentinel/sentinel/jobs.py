"""
SENTINEL-TS :: Background job runner.

A full comparison takes tens of seconds and a cross-dataset benchmark takes
minutes. Holding an HTTP connection open for either is poor practice: the
request can time out at any proxy in between, and the client learns nothing
until it finishes.

So runs are submitted, tracked and collected. A small thread pool executes
them, progress is published as the pipeline advances, and the console polls.
That is also what lets the UI show a real progress bar rather than a decorative
one -- every percentage it displays came from the pipeline itself.

Thread-based rather than process-based because the workload is dominated by
NumPy, SciPy and scikit-learn routines that release the GIL, and because
threads keep results in the same address space with no serialisation cost.
"""

from __future__ import annotations

import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional


@dataclass
class Job:
    id: str
    kind: str
    label: str
    status: str = "queued"  # queued | running | complete | failed
    progress: int = 0
    stage: str = "Queued"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    traceback: Optional[str] = None

    def public(self, include_result: bool = False) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "kind": self.kind,
            "label": self.label,
            "status": self.status,
            "progress": self.progress,
            "stage": self.stage,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }
        if include_result:
            payload["result"] = self.result
        return payload


class JobManager:
    """Submit, track and collect background analyses.

    Completed jobs are retained up to ``max_retained`` so a user can revisit an
    earlier run; beyond that the oldest finished job is evicted. Without a cap,
    a long-lived server accumulates every result bundle it has ever produced.
    """

    def __init__(self, max_workers: int = 2, max_retained: int = 24):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._jobs: Dict[str, Job] = {}
        self._order: list = []
        self._lock = threading.Lock()
        self._max_retained = max_retained

    # -- submission --------------------------------------------------------

    def submit(
        self,
        kind: str,
        label: str,
        fn: Callable[..., Dict[str, Any]],
        **kwargs: Any,
    ) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, label=label)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._evict_locked()
        self._pool.submit(self._run, job, fn, kwargs)
        return job

    def _run(self, job: Job, fn: Callable[..., Dict[str, Any]], kwargs: Dict[str, Any]) -> None:
        job.status = "running"
        job.stage = "Starting"
        job.started_at = datetime.now(timezone.utc).isoformat()

        def progress(stage: str, pct: int) -> None:
            job.stage = stage
            job.progress = int(max(0, min(100, pct)))

        try:
            job.result = fn(progress=progress, **kwargs)
            job.status = "complete"
            job.progress = 100
            job.stage = "Complete"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{type(exc).__name__}: {exc}"
            job.traceback = traceback.format_exc()
            job.stage = "Failed"
        finally:
            job.finished_at = datetime.now(timezone.utc).isoformat()

    # -- retrieval ---------------------------------------------------------

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self) -> list:
        with self._lock:
            return [self._jobs[i].public() for i in reversed(self._order) if i in self._jobs]

    def _evict_locked(self) -> None:
        finished = [
            i
            for i in self._order
            if i in self._jobs and self._jobs[i].status in ("complete", "failed")
        ]
        while len(finished) > self._max_retained:
            victim = finished.pop(0)
            self._jobs.pop(victim, None)
            if victim in self._order:
                self._order.remove(victim)

    def shutdown(self) -> None:  # pragma: no cover
        self._pool.shutdown(wait=False, cancel_futures=True)
