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

Add your Claude API key to `.env` as `ANTHROPIC_API_KEY`. The backend reads it from there or from the environment; the start page tells you whether it found one. Ctrl-C stops both servers. Override ports with `make dev WEB_PORT=3000 SERVER_PORT=9000`.

## Using it

1. Open a JPEG, PNG, or HEIC photo (choose one or drop it on the page).
2. Tell the agent what you want, e.g. "make this look warmer and less washed out, and crop it for Instagram". It streams its reply and lists the edits as it makes them. Each request becomes its own layer.
3. Fine-tune in **Layers**: hide a layer, change its opacity or blend mode, drag any adjustment's slider (light and color sliders preview instantly), add adjustments by hand, or limit a layer to part of the photo with a mask (paint with a brush, or use a linear gradient, radial gradient, or brightness range).
4. **History** lists every agent turn and manual change as a named step. Click one to go back to it; editing from there starts a new branch and keeps the old one. **Undo** / **Redo** (Ctrl/⌘+Z, Shift+Ctrl/⌘+Z) step along the current branch.
5. **Recipes** saves a photo's layers as a named look, to apply to other photos in one click. Crops and painted masks stay with the photo they were made for.
6. Compare with the original by holding **Hold to compare** (or the `\` key) or with **Split view**. Zoom with the mouse wheel and drag to pan.
7. **Download** renders the edits at full resolution as JPEG or PNG. Camera EXIF is kept; GPS location is removed unless you ask to keep it.

Your work is autosaved in the browser: close the tab and the project reopens where you left off, and the start page lists recent projects. The start page has two options: **Choose a photo** to start fresh, or **Open a project** to reopen a saved `.photoagent` file. **Save project** downloads a `.photoagent` file (the original plus every layer and the full history) that you can open again later, here or on another computer.

Photos and their edit histories are kept in `.data/` at the repo root. A downsized preview of the photo is sent to Claude with each request so the agent can see it; the original never leaves your computer. Set `ANTHROPIC_MODEL` in `.env` to use a different Claude model.

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

See [docs/architecture.md](docs/architecture.md) for how the pieces fit together.
