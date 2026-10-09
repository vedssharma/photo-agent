import type { Mask } from '../api/documents'
import {
  MASK_KINDS,
  type MaskTool,
  SEMANTIC_TARGETS,
  type SemanticTarget,
  defaultMask,
  maskChoice,
  maskLabel,
  semanticMask,
} from '../lib/masks'
import { Slider } from './Slider'

interface Props {
  /** What the mask belongs to, for history labels (a layer's name, or "the cutout"). */
  name: string
  mask: Mask | null
  /** The mask cannot be removed (a cutout always keeps something). */
  required?: boolean
  disabled?: boolean
  tool: MaskTool
  onTool: (tool: MaskTool) => void
  /** Commit a new mask (or none) for the layer. */
  onMask: (mask: Mask | null, label: string, coalesce?: string) => void
}

const pct = (v: number) => `${Math.round(v)}%`

/** Choose and tune a mask. Brushes, gradients, and object selections are drawn on the photo. */
export function MaskControls({
  name,
  mask,
  required = false,
  disabled = false,
  tool,
  onTool,
  onMask,
}: Props) {
  function setChoice(choice: string) {
    onTool({ ...tool, picking: false })
    if (choice === '') return onMask(null, `Remove mask from ${name}`)
    if (choice.startsWith('semantic:')) {
      const target = choice.slice('semantic:'.length) as SemanticTarget
      // An object is selected by clicking it, which commits the mask.
      if (target === 'object') return onTool({ ...tool, picking: true })
      return onMask(
        semanticMask(target),
        `Select ${maskLabel('semantic', target).toLowerCase()} for ${name}`,
      )
    }
    const kind = choice as Exclude<Mask['kind'], 'semantic'>
    onMask(defaultMask(kind), `${maskLabel(kind)} mask on ${name}`)
  }

  const picking =
    tool.picking && !(mask?.kind === 'semantic' && mask.target === 'object')
  const choice = picking ? 'semantic:object' : maskChoice(mask)
  const groups = [...new Set(SEMANTIC_TARGETS.map((t) => t.group))]

  function tweak(changes: Partial<Mask>, what: string, key: string) {
    if (!mask) return
    onMask({ ...mask, ...changes } as Mask, `${name} mask ${what}`, key)
  }

  return (
    <fieldset className="mask-controls" disabled={disabled}>
      <label className="field">
        <span>Mask</span>
        <select value={choice} onChange={(e) => setChoice(e.target.value)}>
          {!required && <option value="">None (whole photo)</option>}
          {MASK_KINDS.map((m) => (
            <option key={m.kind} value={m.kind}>
              {m.label}
            </option>
          ))}
          {groups.map((group) => (
            <optgroup key={group} label={group}>
              {SEMANTIC_TARGETS.filter((t) => t.group === group).map((t) => (
                <option key={t.target} value={`semantic:${t.target}`}>
                  {t.label}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
      </label>
      {picking && (
        <p className="hint">
          Click the thing to select on the photo, or drag a box around it.
        </p>
      )}
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
          {mask.kind === 'semantic' && (
            <p className="hint">
              {mask.target === 'object'
                ? 'Click more of it to add to the selection, Shift-click to leave a part out, or drag a new box.'
                : 'Found with AI. Invert to change everything else instead.'}
              {mask.description && ` Selected: ${mask.description}.`}
            </p>
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
