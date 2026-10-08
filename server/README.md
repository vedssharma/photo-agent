# photo-agent server

Python backend (FastAPI), managed with [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run uvicorn photo_agent.main:app --reload --port 8000
```

API docs are served at http://localhost:8000/docs while the server runs.
