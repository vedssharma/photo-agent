from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from photo_agent.vision.backends import Env, Progress, Task, WorkerConfig
from photo_agent.vision.tasks import TASKS
from photo_agent.vision.worker import JobFailedError, ModelWorker


def config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(backends="auto", device="cpu", cache_dir=tmp_path / "models")


def test_inline_worker_runs_jobs_and_records_them(tmp_path: Path) -> None:
    worker = ModelWorker(config(tmp_path), "inline")
    try:
        result = worker.run("self_test", np.full((4, 4, 3), 0.25, np.float32), doc_id="d1")
        assert result.value == pytest.approx(0.25)
        assert result.backend == "self-test"
        (job,) = worker.jobs("d1")
        assert (job.state, job.fraction, job.label) == ("done", 1.0, "Checking the model worker")
        assert worker.jobs("other") == []
    finally:
        worker.close()


def test_failed_jobs_raise(tmp_path: Path) -> None:
    worker = ModelWorker(config(tmp_path), "inline")
    try:
        with pytest.raises(JobFailedError, match="Asked to fail"):
            worker.run("self_test", np.zeros((2, 2, 3), np.float32), {"fail": True})
        with pytest.raises(JobFailedError, match="Unknown task"):
            worker.run("no_such_task")
    finally:
        worker.close()


class Flaky:
    name = "flaky-model"
    license = "test"
    uses_weights = True

    def available(self) -> bool:
        return True

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        raise OSError("could not download weights")


class Steady:
    name = "classical"
    license = "n/a"
    uses_weights = False

    def available(self) -> bool:
        return True

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        progress(0.5, "halfway")
        return "fallback"


def test_a_failing_model_falls_back_and_is_set_aside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    flaky = Flaky()
    task = Task("flaky_task", "Testing", [flaky, Steady()])
    monkeypatch.setitem(TASKS, "flaky_task", task)
    worker = ModelWorker(config(tmp_path), "inline")
    try:
        assert worker.run("flaky_task").value == "fallback"
        assert task.broken == {"flaky-model"}
        assert [b.name for b in task.choices("auto")] == ["classical"]
    finally:
        worker.close()


def test_classical_mode_never_uses_weights() -> None:
    task = Task("flaky_task", "Testing", [Flaky(), Steady()])
    assert [b.name for b in task.choices("classical")] == ["classical"]


def test_process_worker_runs_jobs_in_another_process(tmp_path: Path) -> None:
    worker = ModelWorker(config(tmp_path), "process")
    try:
        result = worker.run("self_test", np.full((3, 3, 3), 0.5, np.float32), doc_id="d")
        assert result.value == pytest.approx(0.5)
        status = worker.status()
        assert status.mode == "process"
        assert status.device == "cpu"
        assert "self_test" in {t.task for t in status.tasks}
    finally:
        worker.close()


def test_models_endpoint_and_document_jobs(client: TestClient, upload: Any) -> None:
    status = client.get("/api/models").json()
    assert status["mode"] == "inline"
    assert all(not t["uses_weights"] for t in status["tasks"])

    doc = upload("landscape.png")
    assert client.get(f"/api/documents/{doc['id']}/jobs").json() == []
    assert client.get("/api/documents/000000000000/jobs").status_code == 404
