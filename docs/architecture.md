# Architecture

photo-agent is a monorepo with two halves:

| Path | What | Stack |
| --- | --- | --- |
| `web/` | Browser app: chat, canvas, layers, history, recipes | Vite, React, TypeScript |
| `server/` | Agent service, edit graph, render engine, model worker | Python, FastAPI, uv |
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

## AI local edits (Phase 3)

```
Browser                                     Backend (server/src/photo_agent)
───────                                     ─────────────────────────────────
MaskControls, MaskOverlay ─ POST …/edits ─▶ masks.py           semantic masks: what to select, plus touch-ups
PhotoCanvas badge ── GET …/jobs ──────────▶ vision/worker.py   model jobs with progress, in their own process
LayersPanel: Cut out, Remove…              vision/selection.py run, cache, and look up model results
LayersPanel: Retouch ── POST …/retouch ───▶ portrait.py        skin, eyes, and teeth layers
Crop & rotate ── POST …/straighten ───────▶ geometry.py        tilt and converging verticals from line segments
GET /api/models                             vision/tasks.py    which backend serves each task
```

**Model worker (`vision/worker.py`, `vision/tasks.py`).** Models run in a separate process (a thread in tests, `MODEL_WORKER=inline`), so a slow or crashing model never blocks the API. Each task (segment the sky, the subject, people, an object; parse faces; inpaint) lists backends best first: an open-source model from the Hugging Face Hub when the optional `models` extra is installed (`uv sync --extra models`), then a classical OpenCV fallback that always works. A model that fails to load or run is set aside and the next backend takes over. Weights download once into `MODEL_DIR` (default `.data/models`) and stay loaded between jobs; `MODEL_DEVICE` picks CUDA, Apple MPS, or CPU. The browser polls `GET …/jobs` while a render waits on the worker and shows the job and its progress over the photo. `GET /api/models` reports which backend serves each task.

| Task | Model (license) | Fallback |
| --- | --- | --- |
| Object at a click or box | SAM 2.1 hiera-small (Apache-2.0) | GrabCut in the box |
| Sky, people | UperNet ConvNeXt on ADE20K (MIT) | color, texture, and position; skin-color faces |
| Main subject, cutouts | BiRefNet (MIT) | GrabCut on the center |
| Face parts | SegFormer face parsing (CelebAMask-HQ, non-commercial) | skin-color face finder with fixed proportions |
| Removal fill | LaMa, ONNX export (Apache-2.0) | multi-scale Telea inpainting with matched grain |

The face-parsing weights are the one non-commercial dependency; swap them before this becomes a product.

**Semantic masks (`masks.py`, `vision/selection.py`).** A new mask kind, `semantic`, stores what to select (a target such as `sky` or `teeth`, or for `object` a box and include/exclude points), never the pixels. The model runs on the framed original at a 1024 px working size; the result is cached in memory and as a PNG under `.data/<doc>/vision/`, keyed by task, backend, framing, and the box and points, and resized to whatever resolution is rendering. So a semantic mask is as editable as a gradient: re-click to change it, invert it, or touch it up. Touch-ups are brush strokes stored on the mask and applied after the model's result (`max(found × kept, painted)`), so painting or erasing never re-runs the model. Recipes keep target masks (the sky is the sky in any photo) but drop object selections and touch-ups.

**Removal (`render.py`).** A layer whose only operation is `remove` fills what its mask selects. Removal layers render before every adjustment layer, wherever they sit in the stack, so later looks apply to the filled photo. The mask is grown by `grow`, the bounding box around it is cropped with margin and sent to the inpainting task, and the patch is cached by the framing and every removal mask up to that one.

**Cutout (`layers.py`, `export.py`).** `EditState.cutout` keeps what its mask selects (the main subject by default) and is applied last. The preview shows a checkerboard; a PNG export carries the alpha channel, and a JPEG export puts the subject on white. A cutout can also have a solid background color.

**Portrait retouching (`portrait.py`).** Two operations, `smooth_skin` (a guided filter with the finest texture added back) and `heal_blemishes` (small dark or red spots, filtered by size and shape, filled from their surroundings), plus a retouch that adds subtle layers masked to skin, eyes, and teeth. Each part stays its own layer to fade or hide.

**Straightening and lens fixes (`geometry.py`).** Two framing operations, `perspective` (stretch the narrow side of converging lines back out to the frame) and `lens_correction` (radial barrel or pincushion undo, scaled so no edge is empty), join crop, rotate, straighten, and flip. Auto straighten runs OpenCV's line segment detector on the framed photo: the tilt most level and plumb lines agree on becomes a `straighten`, and walls on both sides converging at the same rate become a `perspective`. Photos with no clear lines are left alone.

**Agent.** The agent can give any layer a semantic mask (`"brighten just the subject"`), remove things in their own layer, cut out the subject (`cut_out`, `restore_background`), retouch portraits (`retouch_portrait`), and straighten (`auto_straighten`). Its self-check render after each round shows what a selection actually caught, so it can move the box or add points when the model picked the wrong thing. When it adjusts a selection the person touched up, it keeps their strokes.


## Generative edits (Phase 4)

```
Browser                                     Backend (server/src/photo_agent)
───────                                     ─────────────────────────────────
LayersPanel: Generate…, New background…,    generative.py      seeds, models, and cache keys of generative ops
  Relight…, Restyle…, Restore… ─ …/edits ─▶ render.py          content layers, expand, relight, colorize, restyle
Crop & rotate: Expand                       compositing.py     harmonize a subject with a new background
OperationControls: New take, Show options   variants.py        takes on offer, contact sheets
  ── POST/GET …/operations/{op}/options ──▶
RecipesPanel: Add a look ─ POST …/looks/… ▶ looks.py           ready-made looks built from adjustments
ExportDialog: Enlarge ── POST …/export ───▶ export.py          upscaling; credentials.py signs C2PA manifests
(every new prompt) ───────────────────────▶ safety.py          local rules plus a Claude classifier
```

**Generative operations (`operations.py`, `generative.py`).** `generate` (fill a masked area from a prompt), `replace_background`, `relight`, `restore_faces`, `colorize`, and `restyle` are content operations: each stands alone in its layer and, like removals, renders before every adjustment. `expand` is a framing operation, so masks and later crops refer to the expanded frame. Each records its `seed` and `model`, filled in when the edit is made (`generative.stamp`, in the manual-edit route and after each agent tool call), so every render, undo, and export shows the same take, and a new seed gives a different one. The worker tries the recorded model first.

**One result, any size (`vision/selection.py`).** A generative result is made once at a fixed working size (1024 px, 2048 for faces) and cached by everything it depends on except the render size: the framing, every content layer up to it, and the operation without its id or the fields applied afterwards (`harmonize`, `amount`, `options`). Preview and export scale the same patch, so they match, and changing how much of a result to use never regenerates it.

**What each one does (`render.py`).**
- *Generative fill* sends the masked area and its surroundings to SDXL inpainting and blends the patch in through the grown mask.
- *Expand canvas* pads the frame to an aspect ratio (or by side amounts), seeds the new margins with a soft mirror of the photo, and has the model paint them, overlapping the photo slightly so the seam disappears.
- *Background replacement* defaults to everything but the main subject, paints the new scene, composites it through the mask's soft edge, then *harmonizes* (`compositing.py`): the subject's average tone and color shift toward the scene and the scene's light wraps over its edge. Harmonizing needs no model, so its slider is instant.
- *Relighting* redraws the photo under the requested light with IC-Light, then keeps the photo's own fine detail under the new light (`transfer_light`), blended by `amount` and the layer's mask.
- *Face restoration* runs GFPGAN on faces YuNet finds; *colorizing* runs DDColor and keeps the photo's own brightness (`keep_luminance`).
- *Restyle* redraws the whole photo with SDXL at a strength that keeps the composition. Color looks never use it.
- *Upscaling* is an export option (2× or 4×, up to 8000 px across) with Real-ESRGAN, tiled so big photos fit in memory; pixels cross to the worker as 8 bits.

| Task | Model (license) | Fallback |
| --- | --- | --- |
| Fill, expand, new background, restyle | SDXL inpainting 0.1 (CreativeML Open RAIL++-M) | continue the surroundings; a plain studio backdrop; a painterly filter |
| Relight | IC-Light fc on Realistic Vision 5.1, SD 1.5 (CreativeML Open RAIL-M) | shading from the chosen side in the light's color |
| Upscale | Real-ESRGAN x4 (BSD-3-Clause) | Lanczos and a light sharpen |
| Restore faces | GFPGAN 1.4 (Apache-2.0) with YuNet faces (MIT) | denoise and sharpen each face |
| Colorize | DDColor (Apache-2.0) | a hand-tinted look |

The fallbacks are honest stand-ins: the agent is told when no generative model is installed, and the layer says "Made with classical". The new models need the `models` extra, which now includes diffusers, accelerate, and spandrel.

**Looks (`looks.py`).** Ten built-in looks (1970s film, teal and orange, noir, golden hour, …) are layers of ordinary adjustments, including the new `color_grade` (split toning: a hue and amount for shadows, midtones, and highlights). The agent prefers them, or grades by hand, and calls `restyle` only for a new medium ("make it a watercolor").

**Variants (`variants.py`).** "Show me 3 options" puts several seeds on the operation (`options`; the first is the current take) as one history step and generates each. The agent sees them numbered side by side (`show_options`); the layers panel shows them as thumbnails under the seed, and picking one sets the seed, which is instant since each take is cached.

**Safety (`safety.py`).** Every prompt new to the edit state is screened before anything is generated. Local rules always block sexual edits of real people, undressing, and faked documents; with an API key, Claude classifies the rest with a short structured-output call, chiefly for deceptive impersonation of real people. Verdicts are cached; if Claude can't be reached the local rules still apply. A refused manual edit returns 422 with the reason; a refused agent tool call goes back to Claude, and nothing changes.

**Content Credentials (`credentials.py`).** An export in which any generative edit shows is signed with a C2PA manifest (`c2pa-python`) listing each generative edit, its model, and the IPTC `compositeWithTrainedAlgorithmicMedia` source type. The signing certificate and a local certificate authority are made on first use in `.data/c2pa/`. Verifiers read the manifest but report the signer as unknown; a product would sign with a certificate from a C2PA-trusted issuer.
