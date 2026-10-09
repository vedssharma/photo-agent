# photo-agent Roadmap

photo-agent is a web app where anyone can get professional-grade photo edits by describing what they want in plain language. Claude is the agent brain: it reads the photo, plans the edit, and drives a toolbox of image operations and open-source image models. Every edit is non-destructive, so the user can always see, tweak, or undo what the agent did.

## Guiding decisions

These came from the project kickoff (2026-10-08) and shape everything below.

| Topic | Decision |
| --- | --- |
| Audience | General consumers who are not editing experts but want pro-quality results |
| Platform | Web app |
| Interaction | Chat-driven first; suggestions and batch automation come later |
| Edit scope | All kinds: basic adjustments, AI edits, generative edits, style transfer |
| Editing model | Non-destructive, with layers and full history |
| Models | Claude for planning and tool use; open-source image models for pixels |
| Library | Yes: organize, search, tag, cull |
| Formats | JPEG, PNG, HEIC now; RAW later |
| Stack | TypeScript + React frontend, Python backend |
| Business | Personal project for now; keep the door open to a product |

## Architecture at a glance

```
┌──────────────────────────── Browser (React + TS) ────────────────────────────┐
│  Chat panel  │  Canvas / before-after  │  Layers + history  │  Library (later) │
└──────────────────────────────┬────────────────────────────────────────────────┘
                               │ REST + WebSocket (streaming chat, render progress)
┌──────────────────────────────▼──── Python backend (FastAPI) ─────────────────┐
│  Agent service      Claude with tool use; turns requests into edit operations │
│  Edit graph         Document = source image + ordered layers/ops (JSON)       │
│  Render engine      Applies the graph (NumPy/OpenCV/Pillow); proxy + full-res │
│  Model workers      SAM, LaMa, background removal, diffusion, upscalers       │
│  Storage            Local disk in v1; object storage + DB when productized    │
└───────────────────────────────────────────────────────────────────────────────┘
```

Two principles to protect from day one, because they are expensive to retrofit:

1. **The agent never edits pixels directly.** It emits structured operations (`adjust_exposure(+0.4)`, `mask_subject()`, `inpaint(region, prompt)`) that land in the edit graph. That is what makes edits non-destructive, undoable, explainable, and replayable on a full-resolution or RAW source later.
2. **Preview and export are separate renders.** Interactive work happens on a downscaled proxy; export re-renders the same graph at full resolution.

---

## Phase 0: Foundations

**Goal:** a repo anyone can clone and run with one command.

- Monorepo layout: `web/` (Vite + React + TypeScript), `server/` (Python, FastAPI, managed with uv), `docs/`
- Shared API contract (OpenAPI generated from FastAPI, TypeScript client generated from it)
- Lint/format/typecheck: ESLint + Prettier + tsc; Ruff + mypy/pyright
- Test runners: Vitest, pytest; GitHub Actions CI for both
- `.env` handling for the Anthropic API key; a `make dev` (or similar) that starts both halves
- Small fixture set of test photos (JPEG, PNG, HEIC, portrait, landscape, low light)

**Done when:** CI is green on an empty app, and `make dev` serves a hello-world page that talks to the backend.

## Phase 1: v1, chat to edit a single photo

**Goal:** pick a photo from your computer, chat with the agent to edit it, save the result to your computer.

Frontend
- Open a photo via file picker and drag-and-drop (JPEG, PNG, HEIC)
- Canvas with zoom/pan and a before/after toggle (hold-to-compare or split slider)
- Chat panel with streaming responses; the agent explains what it changed in plain words
- Undo / redo of agent turns
- Export: choose format (JPEG/PNG), quality, and download to disk

Backend
- Image ingest: decode with Pillow + `pillow-heif`, honor EXIF orientation, convert to a working color space (sRGB, float32), generate a preview proxy
- **Edit graph v0:** a document is the original image plus an ordered list of operations with parameters. Stored as JSON.
- **Core operation toolbox** (deterministic, fast, no ML):
  - Light: exposure, contrast, highlights, shadows, whites, blacks
  - Color: white balance (temperature/tint), vibrance, saturation, HSL per color band
  - Detail: sharpening, noise reduction, clarity, dehaze
  - Geometry: crop (with aspect presets), rotate, straighten, flip
  - Finishing: vignette, grain, tone curve
- **Agent service:** Claude with tool use, where each operation is a tool. Claude gets the current preview image plus the edit graph, so it can see the photo and reason about what to change. Streams its reply to the UI over WebSocket.
- **Self-check loop:** after applying edits, Claude looks at the new render and corrects obvious overshoots (blown highlights, oversaturation) before replying.
- Export renders the graph at full resolution, preserves EXIF where sensible, strips location by default.

**Done when:** a non-expert can open an iPhone HEIC, type "make this look warmer and less washed out, and crop it for Instagram", see a good result in seconds, undo one step, and download a JPEG.

## Phase 2: Layers, history, and projects

**Goal:** make non-destructive editing visible and controllable, so users trust the agent.

- **Layers:** the edit graph grows from a flat list into a layer stack. Each layer holds operations, an opacity, a blend mode, and an optional mask.
- **Masks v1:** brush, linear gradient, radial gradient, luminosity range
- **History panel:** every agent turn and manual tweak is a named step; click to jump back; branch from any step without losing the other branch
- **Manual controls:** every op the agent applied appears as an editable slider, so users can fine-tune ("the agent got it 90% there")
- **Project files:** save/reopen a project (original + graph) locally; autosave to browser storage
- **Edit recipes:** save a set of ops as a reusable preset and apply it to another photo
- Live slider previews on the client (WebGL shaders for the core adjustments) so dragging is instant; the server stays the source of truth for export

**Done when:** a user can see each change the agent made as its own layer, hide or tweak one, branch history, close the tab, and reopen the project intact.

## Phase 3: AI-powered local edits

**Goal:** edits that understand what is in the photo. This is where consumers start getting results they could not get themselves.

- **Model worker service:** separate Python process (GPU when available, CPU fallback for small models), job queue, progress events to the UI, model weight caching
- **Semantic masks:** Segment Anything (SAM 2) for click-to-select and "select the person / sky / dog"; a dedicated sky segmenter; person and face parsing for portraits. The agent can now target edits: "brighten just the subject", "make the sky more dramatic".
- **Object removal:** LaMa inpainting for distractions, power lines, photobombers
- **Background removal:** BiRefNet or RMBG for cutouts and transparent PNG export
- **Portrait retouching:** skin smoothing that preserves texture, blemish removal, eye/teeth brighten, all mask-driven and subtle by default
- **Auto-straighten and lens correction** (horizon detection, perspective fix)
- Masks produced by models are stored as layer masks, so they stay editable

**Done when:** "remove the person on the left and make the sky bluer" works end to end, and the user can refine the removal mask by brushing.

## Phase 4: Generative edits

**Goal:** add, replace, and reimagine content.

- **Generative fill / inpainting** with a diffusion model (SDXL or Flux inpainting variants): "add a sunset", "replace the trash can with a plant"
- **Outpainting / expand canvas:** extend a photo to a new aspect ratio
- **Background replacement** with lighting harmonization (e.g. IC-Light-style relighting so the subject matches the new scene)
- **Relighting:** change light direction and mood on portraits and products
- **Upscaling / restoration:** Real-ESRGAN or similar for upscaling; face restoration for old or blurry photos; colorize black-and-white
- **Style transfer / looks:** "make it look like 70s film", "Wes Anderson palette" built from color grading first, diffusion-based restyling only when asked
- Each generative op produces its own layer with its prompt, seed, and model recorded, so it can be regenerated or varied ("show me 3 options")
- Safety: block edits that impersonate real people in deceptive contexts, add content credentials (C2PA) on export when generative ops were used

**Done when:** a user can expand a vertical photo to landscape, swap the background, and pick between several generated variants, all as reversible layers.

## Phase 5: A smarter agent

**Goal:** move from "does what you say" to "knows what good looks like".

- **Suggestions mode:** on open, the agent analyzes the photo and proposes 2 to 4 edit directions as thumbnails; the user picks one and refines in chat
- **Plan preview:** for multi-step requests, the agent shows its plan before running expensive or generative steps
- **Photo critique:** explain what is working and what is not (composition, exposure, color), in friendly non-jargon terms, with a one-click fix for each point
- **Personal style memory:** learn the user's taste from accepted and rejected edits; "my usual look"
- **Reference matching:** "make this look like this other photo" (color and tone transfer from a reference image)
- **Eval harness:** a fixed set of photos and prompts with scored outputs, run in CI on agent/prompt changes so quality does not silently regress
- Cost and latency controls: cache vision analyses, use smaller Claude models for routine turns, reserve larger ones for planning and critique

**Done when:** opening a photo and clicking a suggestion produces a result most testers prefer over their own manual edit.

## Phase 6: Library

**Goal:** manage a whole collection, not one photo at a time.

- Import folders or many files at once; thumbnails, grid and filmstrip views
- Albums, star ratings, flags, color labels
- Metadata: EXIF display, date/location/camera filters
- **Auto-tagging** with an open vision model (e.g. CLIP/SigLIP embeddings plus labels) and **natural-language search** ("beach photos from last summer with the dog")
- Face clustering for people albums (opt-in, local-only by default)
- Duplicate and near-duplicate detection
- **Culling:** pick the best shots from a burst or shoot, scoring sharpness, closed eyes, exposure, and aesthetic quality; the agent explains its picks
- **Batch editing:** apply an edit recipe or a chat instruction to a selection ("make all of these consistent"), with per-photo adjustments
- Library database (SQLite locally; Postgres when productized) and a background job queue for indexing

**Done when:** a user can drop in 500 photos from a trip, ask for the best 30, give them a consistent look in one instruction, and export them.

## Phase 7: RAW and color fidelity

**Goal:** serve enthusiasts and get the most out of every image.

- RAW decode via LibRaw (`rawpy`): demosaic, camera profiles, highlight recovery
- 16-bit / float pipeline end to end; wide-gamut working space (linear Rec.2020 or ProPhotoRGB) with proper output transforms
- ICC profile handling and soft-proofing for sRGB / Display P3
- Lens profile corrections (Lensfun)
- HDR merge and panorama stitch
- Export to TIFF and AVIF/WebP in addition to JPEG/PNG

Because the edit graph already separates the source image from the operations, adding RAW is mostly a new decoder and better color math, not a rewrite.

## Phase 8: Productization (only if the project graduates)

**Goal:** turn the personal tool into something others can sign up for.

- Accounts and auth; per-user projects and libraries in cloud storage
- Hosted GPU inference (autoscaling workers, queueing, usage metering)
- Billing and plans (usage-based for generative and GPU-heavy ops)
- Sharing: public links, before/after share cards, export to social sizes
- Mobile-friendly layout and PWA install; camera roll upload
- Privacy: clear data retention, delete-everything, on-device options where feasible
- **Model licensing review:** some open models (for example some Flux variants) forbid commercial use. Track the license of every model from Phase 3 onward so switching to a product does not force a rewrite.
- Observability, error reporting, abuse prevention

---

## Cross-cutting tracks

These run alongside every phase rather than being a phase of their own.

- **Quality:** agent evals (Phase 5) plus golden-image render tests for deterministic ops from Phase 1
- **Performance:** proxy rendering, render caching per layer, progressive previews for slow model jobs
- **UX for non-experts:** plain-language explanations, sensible defaults that err subtle, never a dead end ("I could not do that, here is what I can do instead")
- **Privacy:** photos stay local in the personal-project phases; only previews needed for reasoning are sent to Claude, and that is disclosed

## Open questions to settle as we go

1. **Where models run during the personal phase:** local GPU, a rented GPU box, or a hosted inference API for the heaviest models? Affects Phase 3 and 4 setup. *Phase 3: locally, in a separate worker process on the GPU when there is one (CUDA or Apple MPS) and the CPU otherwise; Phase 3's models are small enough for either, and every task has a classical fallback. Revisit for Phase 4's diffusion models.*
2. **How much of the render engine lives on the client:** WebGL for live core adjustments only (the current plan), or more? *Phase 2 shipped the core-adjustment shaders only; revisit if dragging other sliders feels slow.*
3. **Project file format:** a single bundled file (zip of original + JSON) versus a folder. *Settled in Phase 2: one `.photoagent` zip with `manifest.json`, `document.json`, and the untouched original.*
4. **Which diffusion base** for generative work, weighed on quality, speed, and license.

## Milestone summary

| Phase | Theme | Outcome for the user |
| --- | --- | --- |
| 0 | Foundations | (internal) runnable, tested skeleton |
| 1 | **v1** | Open a photo, chat to edit, download the result |
| 2 | Layers & history | See, tweak, and undo every change; save projects |
| 3 | AI local edits | Select subjects/sky, remove objects, cut out backgrounds |
| 4 | Generative | Fill, expand, replace backgrounds, relight, upscale, restyle |
| 5 | Smarter agent | Suggestions, critique, personal style, reference matching |
| 6 | Library | Organize, search, cull, batch-edit whole shoots |
| 7 | RAW & color | RAW files, 16-bit, wide gamut, HDR/pano |
| 8 | Product | Accounts, hosting, billing, sharing, mobile |
