# photo-agent server

Python backend (FastAPI), managed with [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run uvicorn photo_agent.main:app --reload --port 8000
```

API docs are served at http://localhost:8000/docs while the server runs.

The AI models for local edits (selections, removal, cutouts, face parsing) are an optional extra, since they pull in PyTorch:

```sh
uv sync --extra models
```

Without it, every model task falls back to a classical OpenCV version. `GET /api/models` shows which backend serves each task. Settings (`MODEL_BACKENDS`, `MODEL_DEVICE`, `MODEL_DIR`) are described in `.env.example` at the repo root.
