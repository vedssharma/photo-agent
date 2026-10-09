import type {
  BlendMode,
  DocumentView,
  Layer,
  Operation,
} from '../api/documents'
import type { OperationSpec } from '../api/operations'
import { useState } from 'react'

import type { EditFn } from '../hooks/useManualEdit'
import { type MaskTool, defaultMask, semanticMask } from '../lib/masks'
import { isContentOp, newSeed } from '../lib/operations'
import {
  CUTOUT,
  FRAMING,
  addLayer,
  addOperation,
  moveLayer,
  newLayerId,
  newOpId,
  type Preview,
  removeLayer,
  removeOperation,
  setCutout,
  updateLayer,
  updateOperation,
} from '../lib/state'
import { AddOperation } from './AddOperation'
import { MaskControls } from './MaskControls'
import { OperationControls } from './OperationControls'
import { Slider } from './Slider'

interface Props {
  doc: DocumentView
  /** Records a manual change as a history step. */
  onEdit: EditFn
  /** True while the agent or a previous change is working; changes wait until then. */
  disabled?: boolean
  selected: string | null
  onSelect: (layerId: string | null) => void
  maskTool: MaskTool
  onMaskTool: (tool: MaskTool) => void
  /** Operation specs by name, for the manual controls. */
  specs: Map<string, OperationSpec>
  /** A slider is being dragged, for live previews. */
  onPreview?: (preview: Preview) => void
  /** Adds the portrait retouch layers. */
  onRetouch?: () => void
  /** Levels the photo and squares up converging verticals. */
  onStraighten?: () => void
}

const BLEND_MODES: { value: BlendMode; label: string }[] = [
  { value: 'normal', label: 'Normal' },
  { value: 'luminosity', label: 'Brightness only' },
  { value: 'color', label: 'Color only' },
  { value: 'multiply', label: 'Multiply' },
  { value: 'screen', label: 'Screen' },
  { value: 'overlay', label: 'Overlay' },
  { value: 'soft_light', label: 'Soft light' },
]

type GenerativeTool = 'generate'

type ExpandAspect = Extract<Operation, { op: 'expand' }>['aspect']

const EXPAND_ASPECTS: ExpandAspect[] = [
  '16:9',
  '3:2',
  '4:3',
  '1:1',
  '4:5',
  '9:16',
]

/** What each generative tool asks for. */
const GENERATIVE_TOOLS: Record<
  GenerativeTool,
  { label: string; placeholder: string; submit: string }
> = {
  generate: {
    label: 'What to add',
    placeholder: 'a potted fern, a sunset sky, a red kite…',
    submit: 'Add',
  },
}

/** A layer name from a prompt: its start, capitalized. */
function layerTitle(prompt: string): string {
  const words = prompt.trim().replace(/\s+/g, ' ')
  const short = words.length <= 40 ? words : `${words.slice(0, 39)}…`
  return short.charAt(0).toUpperCase() + short.slice(1)
}

/** Asks for a prompt before a generative layer is added. */
function PromptForm({
  label,
  placeholder,
  submit,
  disabled,
  onSubmit,
  onCancel,
}: {
  label: string
  placeholder: string
  submit: string
  disabled: boolean
  onSubmit: (prompt: string) => void
  onCancel: () => void
}) {
  const [text, setText] = useState('')
  return (
    <form
      className="prompt-form"
      onSubmit={(e) => {
        e.preventDefault()
        if (text.trim()) onSubmit(text.trim())
      }}
    >
      <input
        type="text"
        aria-label={label}
        placeholder={placeholder}
        value={text}
        autoFocus
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Escape') onCancel()
        }}
      />
      <button type="submit" disabled={disabled || !text.trim()}>
        {submit}
      </button>
    </form>
  )
}

const blendLabel = (mode: BlendMode) =>
  BLEND_MODES.find((m) => m.value === mode)?.label ?? mode

/**
 * The layer stack, top first. Each change the agent made is its own layer that can be
 * hidden, faded, re-blended, reordered, or deleted; each such change is a history step.
 */
export function LayersPanel({
  doc,
  onEdit,
  disabled = false,
  selected,
  onSelect,
  maskTool,
  onMaskTool,
  specs,
  onPreview,
  onRetouch,
  onStraighten,
}: Props) {
  const state = doc.state
  const locked = disabled
  const apply = onEdit
  const allSpecs = [...specs.values()]

  function operationControls(op: Operation, layerId: string | null) {
    const spec = specs.get(op.op)
    const opId = op.id ?? ''
    return (
      <OperationControls
        key={opId}
        op={op}
        spec={spec}
        disabled={locked}
        onPreview={
          onPreview &&
          ((param, value) => onPreview({ layerId, opId, param, value }))
        }
        onChange={(changes, label, param) => {
          void apply(
            label,
            (s) => updateOperation(s, opId, changes),
            `op:${opId}:${param}`,
          )
        }}
        onRemove={() =>
          apply(`Remove ${spec?.label ?? op.op}`, (s) =>
            removeOperation(s, opId),
          )
        }
      />
    )
  }

  function add(spec: OperationSpec, layerId: string | null, where: string) {
    const params = Object.fromEntries(
      spec.params.map((p) => [p.name, p.default]),
    )
    const op = { id: newOpId(), op: spec.op, ...params } as Operation
    void apply(`Add ${spec.label} to ${where}`, (s) =>
      addOperation(s, layerId, op),
    )
  }

  const cutout = state.cutout ?? null
  const BACKGROUNDS: { value: string; label: string }[] = [
    { value: '', label: 'Transparent' },
    { value: '#ffffff', label: 'White' },
    { value: '#000000', label: 'Black' },
  ]

  function cutOut() {
    onSelect(CUTOUT)
    void apply('Remove the background', (s) =>
      setCutout(s, {
        visible: true,
        background: null,
        mask: semanticMask('subject'),
      }),
    )
  }

  /** A removal layer, waiting for a click on what to remove. */
  function removeSomething() {
    const layer: Layer = {
      id: newLayerId(),
      name: 'Remove object',
      visible: true,
      opacity: 100,
      blend_mode: 'normal',
      operations: [{ id: newOpId(), op: 'remove', grow: 20 }],
      mask: null,
    }
    onSelect(layer.id)
    onMaskTool({ ...maskTool, picking: true })
    void apply('Remove an object', (s) => addLayer(s, layer))
  }

  /** What a generative tool asks for before it adds its layer, if one is open. */
  const [asking, setAsking] = useState<GenerativeTool | null>(null)

  /** A generative fill layer, waiting for the person to paint where it goes. */
  function generate(prompt: string) {
    const layer: Layer = {
      id: newLayerId(),
      name: layerTitle(prompt),
      visible: true,
      opacity: 100,
      blend_mode: 'normal',
      operations: [
        {
          id: newOpId(),
          op: 'generate',
          prompt,
          seed: newSeed(),
          grow: 10,
          model: '',
        },
      ],
      mask: defaultMask('brush'),
    }
    onSelect(layer.id)
    onMaskTool({ ...maskTool, picking: false, refining: false, erase: false })
    void apply(`Generate “${prompt}”`, (s) => addLayer(s, layer))
  }

  /** Extend the canvas to an aspect ratio, painting new surroundings. */
  function expand(aspect: ExpandAspect) {
    const op = {
      id: newOpId(),
      op: 'expand',
      aspect,
      left: 0,
      right: 0,
      top: 0,
      bottom: 0,
      prompt: '',
      seed: newSeed(),
      model: '',
    } as Operation
    void apply(`Expand canvas to ${aspect}`, (s) => addOperation(s, null, op))
  }

  function newLayer() {
    const layer: Layer = {
      id: newLayerId(),
      name: `Layer ${state.layers.length + 1}`,
      visible: true,
      opacity: 100,
      blend_mode: 'normal',
      operations: [],
      mask: null,
    }
    onSelect(layer.id)
    void apply('New layer', (s) => addLayer(s, layer))
  }

  const change = (layer: Layer, label: string, changes: Partial<Layer>) =>
    apply(
      label,
      (s) => updateLayer(s, layer.id, changes),
      `layer:${layer.id}:${Object.keys(changes).join(',')}`,
    )

  const topFirst = [...state.layers].reverse()
  return (
    <section className="panel layers-panel" aria-label="Layers">
      <div className="panel-head">
        <h2>Layers</h2>
        {!cutout && (
          <button
            type="button"
            className="icon"
            disabled={locked}
            title="Keep the subject and remove the background"
            onClick={cutOut}
          >
            Cut out
          </button>
        )}
        <button
          type="button"
          className="icon"
          disabled={locked}
          title="Remove a person or thing from the photo"
          onClick={removeSomething}
        >
          Remove…
        </button>
        <button
          type="button"
          className="icon"
          disabled={locked}
          aria-expanded={asking === 'generate'}
          title="Paint something new into the photo from a description"
          onClick={() => setAsking(asking === 'generate' ? null : 'generate')}
        >
          Generate…
        </button>
        {onRetouch && (
          <button
            type="button"
            className="icon"
            disabled={locked}
            title="Smooth skin, heal blemishes, and brighten eyes and teeth"
            onClick={onRetouch}
          >
            Retouch
          </button>
        )}
        <button
          type="button"
          className="icon"
          disabled={locked}
          onClick={newLayer}
        >
          + New layer
        </button>
      </div>
      {asking && (
        <PromptForm
          key={asking}
          {...GENERATIVE_TOOLS[asking]}
          disabled={locked}
          onCancel={() => setAsking(null)}
          onSubmit={(prompt) => {
            setAsking(null)
            if (asking === 'generate') generate(prompt)
          }}
        />
      )}
      {state.layers.length === 0 && (
        <p className="empty">
          No edits yet. Each change the agent makes shows up here as a layer.
        </p>
      )}
      <ol className="layer-list">
        {cutout && (
          <li
            className={`layer cutout${selected === CUTOUT ? ' selected' : ''}${cutout.visible ? '' : ' hidden'}`}
          >
            <div className="layer-head">
              <input
                type="checkbox"
                checked={cutout.visible}
                disabled={locked}
                aria-label="Show background removal"
                onChange={(e) =>
                  apply(
                    e.target.checked
                      ? 'Remove the background again'
                      : 'Show the background',
                    (s) =>
                      setCutout(s, (c) => ({
                        ...c,
                        visible: e.target.checked,
                      })),
                  )
                }
              />
              <button
                type="button"
                className="layer-name"
                aria-expanded={selected === CUTOUT}
                onClick={() => onSelect(selected === CUTOUT ? null : CUTOUT)}
              >
                Background removed
              </button>
              <button
                type="button"
                className="icon"
                aria-label="Bring back the background"
                disabled={locked}
                onClick={() => {
                  if (selected === CUTOUT) onSelect(null)
                  void apply('Bring back the background', (s) =>
                    setCutout(s, null),
                  )
                }}
              >
                ✕
              </button>
            </div>
            {selected === CUTOUT && (
              <div className="layer-body">
                <label className="field">
                  <span>Background</span>
                  <select
                    value={
                      BACKGROUNDS.some(
                        (b) => b.value === (cutout.background ?? ''),
                      )
                        ? (cutout.background ?? '')
                        : 'custom'
                    }
                    disabled={locked}
                    onChange={(e) => {
                      const value =
                        e.target.value === 'custom'
                          ? '#808080'
                          : e.target.value || null
                      void apply(
                        `Background: ${e.target.selectedOptions[0]?.text ?? value}`,
                        (s) =>
                          setCutout(s, (c) => ({ ...c, background: value })),
                      )
                    }}
                  >
                    {BACKGROUNDS.map((b) => (
                      <option key={b.value} value={b.value}>
                        {b.label}
                      </option>
                    ))}
                    <option value="custom">Color…</option>
                  </select>
                </label>
                {cutout.background &&
                  !BACKGROUNDS.some((b) => b.value === cutout.background) && (
                    <label className="field">
                      <span>Color</span>
                      <input
                        type="color"
                        value={cutout.background}
                        disabled={locked}
                        onChange={(e) =>
                          apply(
                            `Background color ${e.target.value}`,
                            (s) =>
                              setCutout(s, (c) => ({
                                ...c,
                                background: e.target.value,
                              })),
                            'cutout:color',
                          )
                        }
                      />
                    </label>
                  )}
                <MaskControls
                  name="the cutout"
                  mask={cutout.mask ?? null}
                  required
                  disabled={locked}
                  tool={maskTool}
                  onTool={onMaskTool}
                  onMask={(mask, label, key) =>
                    mask &&
                    apply(
                      label,
                      (s) => setCutout(s, (c) => ({ ...c, mask })),
                      key ? `cutout:mask:${key}` : undefined,
                    )
                  }
                />
              </div>
            )}
          </li>
        )}
        {topFirst.map((layer, i) => {
          const isSelected = layer.id === selected
          const removes = layer.operations.some((op) => op.op === 'remove')
          const content = layer.operations.some(isContentOp)
          const generated = layer.operations.find(
            (op) => 'model' in op && op.model,
          )
          return (
            <li
              key={layer.id}
              className={`layer${isSelected ? ' selected' : ''}${layer.visible ? '' : ' hidden'}`}
            >
              <div className="layer-head">
                <input
                  type="checkbox"
                  checked={layer.visible}
                  disabled={locked}
                  aria-label={`Show ${layer.name}`}
                  onChange={(e) =>
                    change(
                      layer,
                      `${e.target.checked ? 'Show' : 'Hide'} “${layer.name}”`,
                      { visible: e.target.checked },
                    )
                  }
                />
                <button
                  type="button"
                  className="layer-name"
                  aria-expanded={isSelected}
                  onClick={() => onSelect(isSelected ? null : layer.id)}
                >
                  {layer.name}
                </button>
                <button
                  type="button"
                  className="icon"
                  aria-label={`Move ${layer.name} up`}
                  disabled={locked || i === 0}
                  onClick={() =>
                    apply(`Move “${layer.name}” up`, (s) =>
                      moveLayer(s, layer.id, 1),
                    )
                  }
                >
                  ↑
                </button>
                <button
                  type="button"
                  className="icon"
                  aria-label={`Move ${layer.name} down`}
                  disabled={locked || i === topFirst.length - 1}
                  onClick={() =>
                    apply(`Move “${layer.name}” down`, (s) =>
                      moveLayer(s, layer.id, -1),
                    )
                  }
                >
                  ↓
                </button>
                <button
                  type="button"
                  className="icon"
                  aria-label={`Delete ${layer.name}`}
                  disabled={locked}
                  onClick={() => {
                    if (isSelected) onSelect(null)
                    void apply(`Delete layer “${layer.name}”`, (s) =>
                      removeLayer(s, layer.id),
                    )
                  }}
                >
                  ✕
                </button>
              </div>
              {isSelected && (
                <div className="layer-body">
                  <Slider
                    label="Opacity"
                    value={layer.opacity}
                    min={0}
                    max={100}
                    disabled={locked}
                    format={(v) => `${Math.round(v)}%`}
                    onCommit={(opacity) =>
                      change(
                        layer,
                        `“${layer.name}” opacity ${Math.round(opacity)}%`,
                        { opacity },
                      )
                    }
                  />
                  {!content && (
                    <>
                      <label className="field">
                        <span>Blend</span>
                        <select
                          value={layer.blend_mode}
                          disabled={locked}
                          onChange={(e) => {
                            const mode = e.target.value as BlendMode
                            void change(
                              layer,
                              `“${layer.name}” blend: ${blendLabel(mode)}`,
                              { blend_mode: mode },
                            )
                          }}
                        >
                          {BLEND_MODES.map((m) => (
                            <option key={m.value} value={m.value}>
                              {m.label}
                            </option>
                          ))}
                        </select>
                      </label>
                    </>
                  )}
                  {layer.operations.some((op) => op.op === 'generate') && (
                    <p className="hint">
                      Paint where it goes on the photo, or pick a mask below.
                    </p>
                  )}
                  {removes && (
                    <p className="hint">
                      {layer.mask
                        ? 'Removed and filled in from the surroundings. Paint or click to change what is removed.'
                        : 'Choose what to remove: click it on the photo, or pick a mask below.'}
                    </p>
                  )}
                  <MaskControls
                    name={`“${layer.name}”`}
                    mask={layer.mask ?? null}
                    disabled={locked}
                    tool={maskTool}
                    onTool={onMaskTool}
                    onMask={(mask, label, key) =>
                      apply(
                        label,
                        (s) => updateLayer(s, layer.id, { mask }),
                        key ? `layer:${layer.id}:mask:${key}` : undefined,
                      )
                    }
                  />
                  {layer.operations.map((op) =>
                    operationControls(op, layer.id),
                  )}
                  {generated && 'model' in generated && (
                    <p className="hint">
                      Made with {generated.model}
                      {generated.model === 'classical' &&
                        ' (no generative model installed: a rough stand-in)'}
                      .
                    </p>
                  )}
                  {!content && (
                    <AddOperation
                      label="Add adjustment"
                      specs={allSpecs.filter(
                        (s) =>
                          !s.framing &&
                          s.group !== 'retouch' &&
                          s.group !== 'generative',
                      )}
                      disabled={locked}
                      onAdd={(spec) => add(spec, layer.id, `“${layer.name}”`)}
                    />
                  )}
                </div>
              )}
            </li>
          )
        })}
        <li
          className={`layer framing${selected === FRAMING ? ' selected' : ''}`}
        >
          <div className="layer-head">
            <button
              type="button"
              className="layer-name"
              aria-expanded={selected === FRAMING}
              onClick={() => onSelect(selected === FRAMING ? null : FRAMING)}
            >
              Crop &amp; rotate
              {state.framing.length > 0 && ` (${state.framing.length})`}
            </button>
          </div>
          {selected === FRAMING && (
            <div className="layer-body">
              <p className="hint">Applies to the whole photo, before layers.</p>
              {onStraighten && (
                <button
                  type="button"
                  disabled={locked}
                  title="Level the horizon and straighten leaning buildings"
                  onClick={onStraighten}
                >
                  Auto straighten
                </button>
              )}
              <label className="field">
                <span>Expand</span>
                <select
                  value=""
                  disabled={locked}
                  title="Extend the photo past its edges with new surroundings painted to match"
                  onChange={(e) => {
                    const aspect = e.target.value as ExpandAspect
                    if (aspect) expand(aspect)
                  }}
                >
                  <option value="">Expand canvas to…</option>
                  {EXPAND_ASPECTS.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </label>
              {state.framing.map((op) => operationControls(op, null))}
              <AddOperation
                label="Add crop, rotation, or lens fix"
                specs={allSpecs.filter((s) => s.framing)}
                disabled={locked}
                onAdd={(spec) => add(spec, null, 'the framing')}
              />
            </div>
          )}
        </li>
      </ol>
    </section>
  )
}
