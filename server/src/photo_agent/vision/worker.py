"""The model worker: runs model jobs one at a time, off the web server.

By default the worker is a separate process, so loading a model or running one on the CPU
never blocks requests, and a model that crashes takes down only the worker (which restarts
on the next job). Jobs wait in a queue and run in order, which keeps one model's memory in
use at a time, and is what a single GPU wants anyway. Loaded models stay in memory in the
worker between jobs; downloaded weights are cached on disk in the model directory.

Every job reports progress, which the web app shows while it waits (`jobs()`).
"""

from __future__ import annotations

import itertools
import logging
import multiprocessing as mp
import os
import queue
import threading
import time
import traceback
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from photo_agent.vision.backends import Env, WorkerConfig, pick_device

log = logging.getLogger(__name__)

Mode = Literal["process", "inline"]

PREFER = "prefer_backend"
"""A job parameter naming the backend to try first (the model a result was recorded with)."""
JobState = Literal["queued", "running", "done", "failed"]

KEEP_FINISHED_SECONDS = 5.0
"""Finished jobs stay listed briefly, so a poll right after a job still sees it end."""


class JobStatus(BaseModel):
    id: str
    doc_id: str | None
    task: str
    label: str
    """What the person sees, e.g. "Finding the sky"."""
    state: JobState
    fraction: float | None = None
    """Progress from 0 to 1, when the task can tell."""
    message: str = ""
    backend: str | None = None
    """The model (or "classical") that ran it."""


class TaskStatus(BaseModel):
    task: str
    label: str
    backend: str
    """The backend that will run this task."""
    license: str
    uses_weights: bool


class WorkerStatus(BaseModel):
    mode: Mode
    device: str
    tasks: list[TaskStatus]


@dataclass
class JobResult:
    value: Any
    backend: str


class JobFailedError(RuntimeError):
    pass


@dataclass
class _Request:
    job_id: str
    task: str
    image: Any
    params: dict[str, Any]


Send = Callable[[tuple[Any, ...]], None]


def execute(request: _Request, config: WorkerConfig, send: Send) -> None:
    """Run one job, trying the task's backends best first; report through `send`."""
    from photo_agent.vision.tasks import TASKS

    if request.task == "status":
        send(("done", request.job_id, _status(config), "worker"))
        return
    task = TASKS.get(request.task)
    if task is None:
        send(("failed", request.job_id, f"Unknown task {request.task!r}."))
        return
    env = Env(device=_device(config), cache_dir=config.cache_dir)
    error = "No backend can run this task."
    params = dict(request.params)
    choices = task.choices(config.backends)
    prefer = params.pop(PREFER, None)
    if prefer:
        # Re-render with the model that made a result first, when it is still around.
        choices.sort(key=lambda b: b.name != prefer)
    for backend in choices:
        send(("started", request.job_id, backend.name))

        def progress(fraction: float | None, message: str) -> None:
            send(("progress", request.job_id, fraction, message))

        try:
            value = backend.run(request.image, params, progress, env)
        except Exception as exc:
            error = str(exc) or type(exc).__name__
            if not backend.uses_weights:
                break
            # A model that cannot load or run (no download, out of memory) is set aside
            # and the next backend, eventually the classical one, does the job.
            log.warning("Model %s failed, falling back: %s", backend.name, error)
            log.debug("%s", traceback.format_exc())
            task.broken.add(backend.name)
            continue
        send(("done", request.job_id, value, backend.name))
        return
    send(("failed", request.job_id, error))


_DEVICE: dict[WorkerConfig, str] = {}


def _device(config: WorkerConfig) -> str:
    if config not in _DEVICE:
        _DEVICE[config] = pick_device(config.device)
    return _DEVICE[config]


def _status(config: WorkerConfig) -> dict[str, Any]:
    from photo_agent.vision.tasks import TASKS

    tasks = []
    for task in TASKS.values():
        choices = task.choices(config.backends)
        if not choices:
            continue
        best = choices[0]
        tasks.append(
            {
                "task": task.name,
                "label": task.label,
                "backend": best.name,
                "license": best.license,
                "uses_weights": best.uses_weights,
            }
        )
    return {"device": _device(config), "tasks": tasks}


def _serve(requests: Any, responses: Any, config: WorkerConfig) -> None:
    """The worker process's main loop."""
    os.environ.setdefault("HF_HOME", str(config.cache_dir / "huggingface"))
    while True:
        request = requests.get()
        if request is None:
            return
        execute(request, config, responses.put)


@dataclass
class _Job:
    status: JobStatus
    future: Future[JobResult]
    finished_at: float | None = None


class ModelWorker:
    """Submits jobs to the worker and tracks them. Thread-safe."""

    def __init__(self, config: WorkerConfig, mode: Mode = "process") -> None:
        self.config = config
        self.mode = mode
        self._jobs: dict[str, _Job] = {}
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._process: Any = None
        self._requests: Any = None
        self._responses: Any = None
        self._inline: queue.Queue[_Request | None] | None = None

    # Submitting

    def submit(
        self,
        task: str,
        image: Any = None,
        params: dict[str, Any] | None = None,
        *,
        doc_id: str | None = None,
        label: str | None = None,
    ) -> Future[JobResult]:
        from photo_agent.vision.tasks import TASKS

        job_id = f"job{next(self._ids)}"
        known = TASKS.get(task)
        status = JobStatus(
            id=job_id,
            doc_id=doc_id,
            task=task,
            label=label or (known.label if known else task),
            state="queued",
        )
        future: Future[JobResult] = Future()
        request = _Request(job_id, task, image, dict(params or {}))
        with self._lock:
            self._jobs[job_id] = _Job(status, future)
            self._forget_finished()
            self._send(request)
        return future

    def run(
        self,
        task: str,
        image: Any = None,
        params: dict[str, Any] | None = None,
        *,
        doc_id: str | None = None,
        label: str | None = None,
    ) -> JobResult:
        """Submit a job and wait for it. Raises JobFailedError."""
        return self.submit(task, image, params, doc_id=doc_id, label=label).result()

    def jobs(self, doc_id: str | None = None) -> list[JobStatus]:
        """Recent and running jobs, oldest first, optionally for one document."""
        with self._lock:
            self._forget_finished()
            return [
                job.status.model_copy()
                for job in self._jobs.values()
                if doc_id is None or job.status.doc_id == doc_id
            ]

    def status(self) -> WorkerStatus:
        info = self.run("status").value
        return WorkerStatus(mode=self.mode, **info)

    def close(self) -> None:
        with self._lock:
            if self._process is not None:
                self._requests.put(None)
                self._process.join(timeout=5)
                if self._process.is_alive():
                    self._process.kill()
                self._process = None
            if self._inline is not None:
                self._inline.put(None)
                self._inline = None

    # Plumbing

    def _send(self, request: _Request) -> None:
        if self.mode == "inline":
            if self._inline is None:
                self._inline = queue.Queue()
                threading.Thread(target=self._run_inline, args=(self._inline,), daemon=True).start()
            self._inline.put(request)
            return
        if self._process is None or not self._process.is_alive():
            self._start_process()
        self._requests.put(request)

    def _run_inline(self, requests: queue.Queue[_Request | None]) -> None:
        while (request := requests.get()) is not None:
            execute(request, self.config, self._receive)

    def _start_process(self) -> None:
        ctx = mp.get_context("spawn")
        self._requests, self._responses = ctx.Queue(), ctx.Queue()
        self._process = ctx.Process(
            target=_serve,
            args=(self._requests, self._responses, self.config),
            name="photo-agent-models",
            daemon=True,
        )
        self._process.start()
        threading.Thread(
            target=self._listen, args=(self._process, self._responses), daemon=True
        ).start()

    def _listen(self, process: Any, responses: Any) -> None:
        while True:
            try:
                message = responses.get(timeout=0.5)
            except queue.Empty:
                if process.is_alive():
                    continue
                self._fail_unfinished("The model worker stopped unexpectedly.")
                return
            self._receive(message)

    def _fail_unfinished(self, reason: str) -> None:
        with self._lock:
            pending = [j for j in self._jobs.values() if not j.future.done()]
        for job in pending:
            self._receive(("failed", job.status.id, reason))

    def _receive(self, message: tuple[Any, ...]) -> None:
        kind, job_id, *rest = message
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.future.done():
                return
            status = job.status
            if kind == "started":
                status.state, status.backend = "running", rest[0]
                status.fraction, status.message = None, ""
            elif kind == "progress":
                status.state = "running"
                status.fraction, status.message = rest[0], rest[1]
            elif kind == "done":
                status.state, status.fraction, status.backend = "done", 1.0, rest[1]
                job.finished_at = time.monotonic()
            elif kind == "failed":
                status.state, status.message = "failed", rest[0]
                job.finished_at = time.monotonic()
        if kind == "done":
            job.future.set_result(JobResult(rest[0], rest[1]))
        elif kind == "failed":
            job.future.set_exception(JobFailedError(rest[0]))

    def _forget_finished(self) -> None:
        cutoff = time.monotonic() - KEEP_FINISHED_SECONDS
        for job_id in [
            j.status.id
            for j in self._jobs.values()
            if j.finished_at is not None and j.finished_at < cutoff
        ]:
            del self._jobs[job_id]
