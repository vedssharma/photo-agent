"""The tasks the model worker can run, each with its backends in order of preference."""

from __future__ import annotations

from photo_agent.vision.backends import SelfTest, Task

TASKS: dict[str, Task] = {
    task.name: task
    for task in [
        Task("self_test", "Checking the model worker", [SelfTest()]),
    ]
}
