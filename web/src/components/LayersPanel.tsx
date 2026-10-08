import { useState } from 'react'

import {
  type BlendMode,
  type DocumentView,
  type EditState,
  type Layer,
  editByHand,
} from '../api/documents'
import { opSummary } from '../lib/operations'
import { moveLayer, removeLayer, updateLayer } from '../lib/state'
import { Slider } from './Slider'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  /** True while the agent is editing; manual changes wait until it finishes. */
  disabled?: boolean
  selected: string | null
  onSelect: (layerId: string | null) => void
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
  onDocument,
  disabled = false,
  selected,
  onSelect,
}: Props) {
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const state = doc.state
  const locked = disabled || working

  async function apply(label: string, next: EditState, coalesce?: string) {
    setWorking(true)
    setError(null)
    try {
      onDocument(await editByHand(doc.id, { label, state: next, coalesce }))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  const change = (layer: Layer, label: string, changes: Partial<Layer>) =>
    apply(
      label,
      updateLayer(state, layer.id, changes),
      `layer:${layer.id}:${Object.keys(changes).join(',')}`,
    )

  const topFirst = [...state.layers].reverse()
  return (
    <section className="panel layers-panel" aria-label="Layers">
      <h2>Layers</h2>
      {state.layers.length === 0 && state.framing.length === 0 && (
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
                    apply(
                      `Move “${layer.name}” up`,
                      moveLayer(state, layer.id, 1),
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
                    apply(
                      `Move “${layer.name}” down`,
                      moveLayer(state, layer.id, -1),
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
                    void apply(
                      `Delete layer “${layer.name}”`,
                      removeLayer(state, layer.id),
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
                  <ul className="ops">
                    {layer.operations.map((op, j) => (
                      <li key={op.id ?? j}>{opSummary(op)}</li>
                    ))}
                  </ul>
                </div>
              )}
            </li>
          )
        })}
        {state.framing.length > 0 && (
          <li className="layer framing">
            <div className="layer-head">
              <span className="layer-name">Crop &amp; rotate</span>
            </div>
            <ul className="ops">
              {state.framing.map((op, j) => (
                <li key={op.id ?? j}>{opSummary(op)}</li>
              ))}
            </ul>
          </li>
        )}
      </ol>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
