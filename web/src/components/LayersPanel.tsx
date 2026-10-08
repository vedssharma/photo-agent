import type {
  BlendMode,
  DocumentView,
  Layer,
  Operation,
} from '../api/documents'
import type { OperationSpec } from '../api/operations'
import type { EditFn } from '../hooks/useManualEdit'
import type { MaskTool } from '../lib/masks'
import {
  FRAMING,
  addLayer,
  addOperation,
  moveLayer,
  newLayerId,
  newOpId,
  type Preview,
  removeLayer,
  removeOperation,
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
  /** A slider is being dragged (or released, with null), for live previews. */
  onPreview?: (preview: Preview | null) => void
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
          onPreview?.(null)
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
        <button
          type="button"
          className="icon"
          disabled={locked}
          onClick={newLayer}
        >
          + New layer
        </button>
      </div>
      {state.layers.length === 0 && (
        <p className="empty">
          No edits yet. Each change the agent makes shows up here as a layer.
        </p>
      )}
      <ol className="layer-list">
        {topFirst.map((layer, i) => {
          const isSelected = layer.id === selected
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
                  <MaskControls
                    layer={layer}
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
                  <AddOperation
                    label="Add adjustment"
                    specs={allSpecs.filter((s) => !s.framing)}
                    disabled={locked}
                    onAdd={(spec) => add(spec, layer.id, `“${layer.name}”`)}
                  />
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
              {state.framing.map((op) => operationControls(op, null))}
              <AddOperation
                label="Add crop or rotation"
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
