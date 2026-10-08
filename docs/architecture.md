# Architecture

photo-agent is a monorepo with two halves:

| Path | What | Stack |
| --- | --- | --- |
| `web/` | Browser app: chat, canvas, layers, history, recipes | Vite, React, TypeScript |
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

## Layers, history, and projects (Phase 2)

```
Browser                                    Backend (server/src/photo_agent)
───────                                    ─────────────────────────────────
LayersPanel, MaskOverlay ─ POST …/edits ─▶ graph.py      record a manual change as a named step
  (sliders: GET /api/operations)           controls.py   parameter ranges for every operation
LivePreview (WebGL)                        render.py     framing, then each layer blended in
HistoryPanel ── POST …/checkout ─────────▶ graph.py      jump to any step; new edits branch
RecipesPanel ── /api/recipes, …/recipes/… ▶ recipes.py   saved looks in .data/recipes.json
Save project ── GET  …/project ──────────▶ projects.py   .photoagent zip
useAutosave  ── GET  …/graph, …/source     (browser keeps a copy in IndexedDB)
PhotoPicker  ── POST /api/projects ──────▶ projects.py   reopen a zip, or restore from the browser copy
```

**Edit state (`layers.py`, `masks.py`).** The state is the framing (crop, rotation, flips) plus a stack of layers. Each layer has adjustment operations, visibility, opacity, a blend mode, and an optional mask. Rendering applies the framing first, then each visible layer bottom to top: the layer's operations run on the result so far, and the output is blended back by blend mode and weighted by opacity times the mask. Masks are in fractions of the framed photo, so they line up at any resolution. Brush strokes are vector data, rasterized at render time; gradient and brightness-range masks are parametric.

**History (`graph.py`).** The document is a tree of steps. Each step stores the full edit state after it, its parent, a name, and whether the agent or a person made it. `head` is the step on screen; `tip` is where redo leads. Editing from an earlier step adds a sibling branch, and nothing is ever deleted. Manual tweaks to the same control (one slider drag after another) fold into one step via a `coalesce` key. Phase 1 documents (a flat turn list) are migrated on load.

**Agent.** The agent groups each distinct change into its own named layer (`add_layer`), can update or remove layers and give them gradient or brightness-range masks, and still has the per-operation tools. Operations it adds without first adding a layer go into one named after the request. It sees the current framing and layer stack with every request, so it respects what the person changed by hand.

**Manual controls and live previews.** `GET /api/operations` describes every operation's parameters, so the sliders are generated, not hand-built. A slider commits on release. While it is dragged, `LivePreview` applies the change on top of the current server preview with a WebGL shader that mirrors `render.py` for the core light and color adjustments (with a 2D canvas fallback). The server render replaces it once it arrives and stays the source of truth.

**Projects (`projects.py`).** A `.photoagent` file is a zip of `manifest.json`, `document.json` (the step tree and chat), and the untouched original. Reopening keeps the document id when it is free. The browser autosaves the graph and the original to IndexedDB after each change (up to 8 projects) and remembers the open project, so a closed tab reopens where it was, even if the server's data folder was cleared.

**Recipes (`recipes.py`).** A recipe is a named copy of a document's layers. Saving drops brush masks, since they only fit the photo they were painted on; framing is never part of a recipe. Applying one adds its layers on top, with fresh ids, as one manual step.
