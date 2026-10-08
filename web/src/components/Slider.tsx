import { useState } from 'react'

interface Props {
  label: string
  value: number
  min: number
  max: number
  step?: number
  disabled?: boolean
  /** Shows the value next to the slider. */
  format?: (value: number) => string
  /** Called continuously while dragging, for live previews. */
  onInput?: (value: number) => void
  /** Called once when the user lets go (or finishes with the keyboard). */
  onCommit: (value: number) => void
}

/**
 * A labeled range slider that reports every movement through `onInput` but commits only
 * when the user lets go, so dragging does not flood the history with steps.
 */
export function Slider({
  label,
  value,
  min,
  max,
  step = 1,
  disabled = false,
  format = String,
  onInput,
  onCommit,
}: Props) {
  const [draft, setDraft] = useState<number | null>(null)
  const shown = draft ?? value

  function commit() {
    if (draft === null) return
    setDraft(null)
    if (draft !== value) onCommit(draft)
  }

  return (
    <label className="slider">
      <span className="slider-label" aria-hidden="true">
        {label}
      </span>
      <input
        type="range"
        aria-label={label}
        min={min}
        max={max}
        step={step}
        value={shown}
        disabled={disabled}
        onChange={(e) => {
          const next = Number(e.target.value)
          setDraft(next)
          onInput?.(next)
        }}
        onPointerUp={commit}
        onKeyUp={commit}
        onBlur={commit}
      />
      <output className="slider-value">{format(shown)}</output>
    </label>
  )
}
