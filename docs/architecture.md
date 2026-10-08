# Architecture

photo-agent is a monorepo with two halves:

| Path | What | Stack |
| --- | --- | --- |
| `web/` | Browser app: chat, canvas, layers, library | Vite, React, TypeScript |
| `server/` | Agent service, edit graph, render engine, model workers | Python, FastAPI, uv |
| `api/` | OpenAPI contract, generated from the server; the web client types are generated from it | JSON |
| `docs/` | Design notes | Markdown |

In development the Vite dev server (port 5173) proxies every `/api/*` request to the FastAPI server (port 8000), so the browser only ever talks to one origin.

See [`../ROADMAP.md`](../ROADMAP.md) for the phased plan and the two core principles (the agent emits structured operations rather than editing pixels; preview and export are separate renders).
