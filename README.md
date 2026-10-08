# photo-agent

Agentic photo editing: describe the edit you want in plain language, and Claude plans it and drives a toolbox of image operations and open-source image models. Every edit is non-destructive.

See [ROADMAP.md](ROADMAP.md) for the plan and [docs/](docs/) for design notes.

## Quick start

Prerequisites: [uv](https://docs.astral.sh/uv/getting-started/installation/), Node.js 22+, and `make`.

```sh
make dev
```

That installs dependencies, creates `.env` from `.env.example` on first run, and starts both halves:

- Web app: http://localhost:5173
- API docs: http://localhost:8000/docs

Add your Claude API key to `.env` as `ANTHROPIC_API_KEY`. The backend reads it from there or from the environment; the hello page tells you whether it found one. Ctrl-C stops both servers. Override ports with `make dev WEB_PORT=3000 SERVER_PORT=9000`.

## Common tasks

| Command | What it does |
| --- | --- |
| `make dev` | Run the backend and web app with hot reload |
| `make check` | Lint, format check, typecheck, and test both halves (what CI runs) |
| `make format` | Auto-format and auto-fix lint in both halves |
| `make test` | Run pytest and Vitest |
| `make api` | Regenerate the OpenAPI contract and TypeScript client types after changing backend routes |

## Layout

```
web/      Vite + React + TypeScript frontend
server/   Python (FastAPI) backend, managed with uv
api/      OpenAPI contract shared by web and server (generated)
docs/     Design notes
fixtures/ Test photos (JPEG, PNG, HEIC; portrait, landscape, low light)
```
