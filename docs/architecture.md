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

## The v1 editor (Phase 1)

```
Browser                                   Backend (server/src/photo_agent)
───────                                   ─────────────────────────────────
PhotoPicker ── POST /api/documents ─────▶ routes.py → store.py   decode (imaging.py), save original + document.json
PhotoCanvas ◀─ GET  …/preview, …/before ─ render.py              apply the operation list to the preview proxy
ChatPanel  ◀─▶ WS   …/chat ─────────────▶ agent.py               Claude tool loop; each tool adds an operation
HistoryButtons ─ POST …/undo, …/redo ───▶ graph.py               move the cursor over recorded turns
ExportDialog ── POST …/export ──────────▶ export.py              full-resolution render + EXIF (GPS stripped)
```

**Edit graph (`graph.py`, `operations.py`).** A document is the original file plus an ordered list of operations, stored as JSON. Each operation is a typed Pydantic model (21 kinds: light, color, detail, geometry, finishing). Every agent turn that changes the photo records the full operation list after it, so undo and redo move a cursor over turns. A separate chat log keeps the conversation linear and notes undo/redo events so the agent knows when its earlier change was taken back.

**Render engine (`render.py`).** Applies operations in order to float32 sRGB pixels with NumPy and OpenCV. Spatial parameters are relative to the original image, so the preview proxy (long edge 1600 px) and the full-resolution export look the same. Golden-image tests in `server/tests/golden/` pin each operation's look; regenerate them with `UPDATE_GOLDEN=1 uv run pytest tests/test_render.py` after an intentional change.

**Agent (`agent.py`, `diagnostics.py`).** Each operation is a Claude tool, plus `update_operation` and `remove_operation`. Every request includes a downsized render and the current operation list. After each round of tool calls the agent sees the new render and measurements against the original (blown highlights, crushed shadows, oversaturation), so it can correct overshoots before it replies. The reply streams to the browser over the chat WebSocket; the event types (`ChatEvent`) are in the OpenAPI schema so the web client gets generated types for them.

**Export (`export.py`).** Re-renders the graph on the full-resolution original, embeds an sRGB profile, keeps EXIF with orientation reset and size updated, and removes GPS location by default.
