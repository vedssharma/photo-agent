import type { Layer, Mask, MaskKind } from '../api/documents'
import { MASK_KINDS, type MaskTool, defaultMask, maskLabel } from '../lib/masks'
import { Slider } from './Slider'

interface Props {
  layer: Layer
  disabled?: boolean
  tool: MaskTool
  onTool: (tool: MaskTool) => void
  /** Commit a new mask (or none) for the layer. */
  onMask: (mask: Mask | null, label: string, coalesce?: string) => void
}

const pct = (v: number) => `${Math.round(v)}%`

/** Choose and tune the selected layer's mask. Brushes and gradients are drawn on the photo. */
export function MaskControls({
  layer,
  disabled = false,
  tool,
  onTool,
  onMask,
}: Props) {
  const mask = layer.mask ?? null
  const name = `“${layer.name}”`

  function setKind(kind: MaskKind | '') {
    if (kind === '') onMask(null, `Remove mask from ${name}`)
    else onMask(defaultMask(kind), `${maskLabel(kind)} mask on ${name}`)
  }

  function tweak(changes: Partial<Mask>, what: string, key: string) {
    if (!mask) return
    onMask({ ...mask, ...changes } as Mask, `${name} mask ${what}`, key)
  }

  return (
    <fieldset className="mask-controls" disabled={disabled}>
      <label className="field">
        <span>Mask</span>
        <select
          value={mask?.kind ?? ''}
          onChange={(e) => setKind(e.target.value as MaskKind | '')}
        >
          <option value="">None (whole photo)</option>
          {MASK_KINDS.map((m) => (
            <option key={m.kind} value={m.kind}>
              {m.label}
            </option>
          ))}
        </select>
      </label>
      {mask && (
        <>
          <div className="checks">
            <label>
              <input
                type="checkbox"
                checked={mask.invert}
                onChange={(e) =>
                  tweak(
                    { invert: e.target.checked },
                    e.target.checked ? 'inverted' : 'not inverted',
                    'invert',
                  )
                }
              />
              Invert
            </label>
            <label>
              <input
                type="checkbox"
                checked={tool.show}
                onChange={(e) => onTool({ ...tool, show: e.target.checked })}
              />
              Show mask
            </label>
          </div>
          {mask.kind === 'brush' && (
            <>
              <p className="hint">
                Paint on the photo where this layer should apply.
              </p>
              <div className="segmented" role="group" aria-label="Brush mode">
                <button
                  type="button"
                  aria-pressed={!tool.erase}
                  onClick={() => onTool({ ...tool, erase: false })}
                >
                  Paint
                </button>
                <button
                  type="button"
                  aria-pressed={tool.erase}
                  onClick={() => onTool({ ...tool, erase: true })}
                >
                  Erase
                </button>
                <button
                  type="button"
                  disabled={mask.strokes.length === 0}
                  onClick={() => tweak({ strokes: [] }, 'cleared', 'clear')}
                >
                  Clear
                </button>
              </div>
              <Slider
                label="Brush size"
                value={tool.size * 100}
                min={0.5}
                max={25}
                step={0.5}
                format={(v) => `${v}%`}
                onCommit={(v) => onTool({ ...tool, size: v / 100 })}
              />
              <Slider
                label="Hardness"
                value={tool.hardness}
                min={0}
                max={100}
                format={pct}
                onCommit={(hardness) => onTool({ ...tool, hardness })}
              />
            </>
          )}
          {mask.kind === 'linear' && (
            <p className="hint">
              Drag on the photo from where the effect is full to where it fades
              out.
            </p>
          )}
          {mask.kind === 'radial' && (
            <>
              <p className="hint">
                Drag on the photo to draw the ellipse; drag its center to move
                it.
              </p>
              <Slider
                label="Feather"
                value={mask.feather}
                min={0}
                max={100}
                format={pct}
                onCommit={(feather) =>
                  tweak(
                    { feather },
                    `feather ${Math.round(feather)}`,
                    'feather',
                  )
                }
              />
            </>
          )}
          {mask.kind === 'luminosity' && (
            <>
              <Slider
                label="From"
                value={mask.low * 100}
                min={0}
                max={100}
                format={pct}
                onCommit={(v) =>
                  tweak(
                    { low: Math.min(v / 100, mask.high) },
                    `from ${Math.round(v)}%`,
                    'low',
                  )
                }
              />
              <Slider
                label="To"
                value={mask.high * 100}
                min={0}
                max={100}
                format={pct}
                onCommit={(v) =>
                  tweak(
                    { high: Math.max(v / 100, mask.low) },
                    `to ${Math.round(v)}%`,
                    'high',
                  )
                }
              />
              <Slider
                label="Softness"
                value={mask.feather * 100}
                min={0}
                max={50}
                format={pct}
                onCommit={(v) =>
                  tweak(
                    { feather: v / 100 },
                    `softness ${Math.round(v)}%`,
                    'feather',
                  )
                }
              />
            </>
          )}
        </>
      )}
    </fieldset>
  )
}
