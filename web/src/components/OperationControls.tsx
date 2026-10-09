import { useState } from 'react'

import type { Operation } from '../api/documents'
import type { OperationSpec, ParamSpec } from '../api/operations'
import { formatValue, newSeed, opSummary } from '../lib/operations'
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
  /** Offers several takes on a generative operation to pick from. */
  onOptions?: () => void
  /** Preview URL of one take, by seed. */
  optionUrl?: (seed: number) => string
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
  onOptions,
  optionUrl,
}: Props) {
  const values = op as unknown as Record<string, unknown>
  const title = spec?.label ?? opSummary(op)
  const numeric = spec?.params.filter((p) => p.kind === 'number') ?? []
  const options = Array.isArray(values.options)
    ? (values.options as number[])
    : []

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
        if (param.kind === 'text')
          return (
            <TextParam
              key={param.name}
              label={param.label}
              value={typeof value === 'string' ? value : ''}
              disabled={disabled}
              onCommit={(text) =>
                onChange(
                  { [param.name]: text },
                  `${title}: “${text}”`,
                  param.name,
                )
              }
            />
          )
        if (param.kind === 'seed')
          return (
            <div className="field" key={param.name}>
              <span>{param.label}</span>
              <span className="seed">
                <code>{typeof value === 'number' ? value : '…'}</code>
                <button
                  type="button"
                  className="icon"
                  disabled={disabled}
                  title="Generate it again with a new seed"
                  onClick={() =>
                    onChange(
                      { [param.name]: newSeed() },
                      `${title}: new take`,
                      param.name,
                    )
                  }
                >
                  New take
                </button>
                {onOptions && (
                  <button
                    type="button"
                    className="icon"
                    disabled={disabled}
                    title="Generate a few takes to compare"
                    onClick={onOptions}
                  >
                    Show options
                  </button>
                )}
              </span>
              {optionUrl && options.length > 1 && (
                <div className="options" role="group" aria-label="Takes">
                  {options.map((seed, i) => (
                    <button
                      key={seed}
                      type="button"
                      className={seed === value ? 'option picked' : 'option'}
                      aria-pressed={seed === value}
                      aria-label={`Take ${i + 1}`}
                      disabled={disabled}
                      onClick={() =>
                        onChange(
                          { [param.name]: seed },
                          `${title}: take ${i + 1}`,
                          param.name,
                        )
                      }
                    >
                      <img src={optionUrl(seed)} alt="" loading="lazy" />
                      <span>{i + 1}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
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

/** A line of text, such as what to generate, committed on Enter or when leaving the field. */
function TextParam({
  label,
  value,
  disabled,
  onCommit,
}: {
  label: string
  value: string
  disabled: boolean
  onCommit: (text: string) => void
}) {
  const [draft, setDraft] = useState(value)
  const [shown, setShown] = useState(value)
  if (shown !== value) {
    // The value changed elsewhere (undo, the agent); show it.
    setShown(value)
    setDraft(value)
  }
  const commit = () => {
    const text = draft.trim()
    if (text && text !== value) onCommit(text)
    else setDraft(value)
  }
  return (
    <label className="field">
      <span>{label}</span>
      <input
        type="text"
        value={draft}
        disabled={disabled}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === 'Enter') commit()
        }}
      />
    </label>
  )
}
