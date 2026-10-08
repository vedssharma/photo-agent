import type { Operation } from '../api/documents'
import type { OperationSpec, ParamSpec } from '../api/operations'
import { formatValue, opSummary } from '../lib/operations'
import { Slider } from './Slider'

interface Props {
  op: Operation
  spec: OperationSpec | undefined
  disabled?: boolean
  /** Commit a parameter change. `label` names it for the history. */
  onChange: (
    changes: Record<string, unknown>,
    label: string,
    param: string,
  ) => void
  /** Live value while a slider is dragged, for instant previews. */
  onPreview?: (param: string, value: number) => void
  onRemove: () => void
}

/**
 * Sliders and pickers for every parameter of one operation, so people can fine-tune what
 * the agent did.
 */
export function OperationControls({
  op,
  spec,
  disabled = false,
  onChange,
  onPreview,
  onRemove,
}: Props) {
  const values = op as unknown as Record<string, unknown>
  const title = spec?.label ?? opSummary(op)
  const numeric = spec?.params.filter((p) => p.kind === 'number') ?? []

  function label(param: ParamSpec, text: string) {
    return numeric.length === 1 && param.kind === 'number'
      ? `${title} ${text}`
      : `${title}: ${param.label.toLowerCase()} ${text}`
  }

  return (
    <div className="op-controls">
      <div className="op-head">
        <span className="op-title" title={spec?.description}>
          {title}
        </span>
        <button
          type="button"
          className="icon"
          aria-label={`Remove ${title}`}
          disabled={disabled}
          onClick={onRemove}
        >
          ✕
        </button>
      </div>
      {spec?.params.map((param) => {
        const value = values[param.name]
        if (param.kind === 'number')
          return (
            <Slider
              key={param.name}
              label={param.label}
              value={typeof value === 'number' ? value : 0}
              min={param.min ?? 0}
              max={param.max ?? 100}
              step={param.step ?? 1}
              disabled={disabled}
              format={(v) => formatValue(param, v)}
              onInput={onPreview && ((v) => onPreview(param.name, v))}
              onCommit={(v) =>
                onChange(
                  { [param.name]: v },
                  label(param, formatValue(param, v)),
                  param.name,
                )
              }
            />
          )
        if (param.kind === 'choice')
          return (
            <label className="field" key={param.name}>
              <span>{param.label}</span>
              <select
                value={String(value)}
                disabled={disabled}
                onChange={(e) => {
                  const choice =
                    param.choices?.find((c) => String(c) === e.target.value) ??
                    e.target.value
                  onChange(
                    { [param.name]: choice },
                    label(param, String(choice)),
                    param.name,
                  )
                }}
              >
                {param.choices?.map((c) => (
                  <option key={String(c)} value={String(c)}>
                    {String(c)}
                  </option>
                ))}
              </select>
            </label>
          )
        return (
          <p className="hint" key={param.name}>
            {param.label}: {Array.isArray(value) ? value.length : 0} points (ask
            the agent to reshape it)
          </p>
        )
      })}
    </div>
  )
}
