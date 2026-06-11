"""A single-worker background job queue.

All heavy local work (folder ingest, folder scan, export) runs here instead of
ad-hoc threads. One worker processes a FIFO queue, so jobs serialize cleanly
(they all contend for the GPU/CPU) and a new submission *queues* behind a
running one rather than being rejected. The UI polls list_jobs() for a live
status indicator.

A job is a callable ``fn(progress: dict, cancel: threading.Event) -> result``.
It should update ``progress`` in place — typically ``phase``/``done``/``total`` —
and may check ``cancel.is_set()`` to stop early.
"""
from __future__ import annotations

import itertools
import logging
import queue
import threading
import time

log = logging.getLogger("aesthetically.jobs")

TERMINAL = {"done", "failed", "cancelled"}


class Job:
    def __init__(self, job_id: int, kind: str, label: str, fn):
        self.id = job_id
        self.kind = kind                 # 'ingest' | 'scan' | 'export'
        self.label = label               # human text, e.g. the folder name
        self.fn = fn
        self.state = "queued"            # queued | running | done | failed | cancelled
        self.progress: dict = {"phase": "queued", "done": 0, "total": 0}
        self.result = None
        self.error: str | None = None
        self.created = time.time()
        self.started: float | None = None
        self.finished: float | None = None
        self.cancel = threading.Event()

    def as_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "label": self.label,
            "state": self.state, "phase": self.progress.get("phase"),
            "done": self.progress.get("done", 0), "total": self.progress.get("total", 0),
            "error": self.error, "result": self.result,
            "created": self.created, "started": self.started, "finished": self.finished,
        }


class JobManager:
    def __init__(self, keep: int = 50):
        self._q: queue.Queue[int] = queue.Queue()
        self._jobs: dict[int, Job] = {}
        self._order: list[int] = []
        self._counter = itertools.count(1)
        self._keep = keep
        self._lock = threading.Lock()
        self._worker = threading.Thread(target=self._run, daemon=True, name="job-worker")
        self._worker.start()

    def submit(self, kind: str, label: str, fn) -> Job:
        with self._lock:
            job = Job(next(self._counter), kind, label, fn)
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim()
        self._q.put(job.id)
        return job

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [self._jobs[i].as_dict() for i in self._order]

    def get(self, job_id: int) -> Job | None:
        return self._jobs.get(job_id)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.state in ("queued", "running"))

    def latest(self, kind: str) -> Job | None:
        with self._lock:
            for i in reversed(self._order):
                if self._jobs[i].kind == kind:
                    return self._jobs[i]
        return None

    def cancel(self, job_id: int) -> bool:
        job = self._jobs.get(job_id)
        if not job or job.state in TERMINAL:
            return False
        job.cancel.set()
        if job.state == "queued":      # not yet picked up — mark immediately
            job.state = "cancelled"
            job.finished = time.time()
        return True

    def _trim(self) -> None:
        finished = [i for i in self._order if self._jobs[i].state in TERMINAL]
        while len(self._order) > self._keep and finished:
            drop = finished.pop(0)
            self._order.remove(drop)
            self._jobs.pop(drop, None)

    def _run(self) -> None:
        while True:
            job_id = self._q.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.cancel.is_set():
                job.state = "cancelled"
                job.finished = job.finished or time.time()
                continue
            job.state = "running"
            job.started = time.time()
            job.progress["phase"] = "running"
            try:
                job.result = job.fn(job.progress, job.cancel)
                job.state = "cancelled" if job.cancel.is_set() else "done"
            except Exception as e:  # noqa: BLE001 — record any failure, keep worker alive
                log.exception("job %s (%s) failed", job.id, job.kind)
                job.error = f"{type(e).__name__}: {e}"
                job.state = "failed"
            finally:
                job.finished = time.time()


# process-wide singleton
manager = JobManager()
