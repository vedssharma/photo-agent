import { useEffect, useRef, useState } from 'react'

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
  /** Called once when the user lets go (or pauses after using the keyboard). */
  onCommit: (value: number) => void
}

/** How long to wait after the last arrow key before committing. */
const KEY_PAUSE_MS = 400

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
  // After letting go, keep showing the new value until the saved one comes back.
  const [committed, setCommitted] = useState<number | null>(null)
  const [seen, setSeen] = useState(value)
  if (value !== seen) {
    setSeen(value)
    setCommitted(null)
  }
  const shown = draft ?? committed ?? value
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  useEffect(
    () => () => {
      if (timer.current) clearTimeout(timer.current)
    },
    [],
  )

  function commit() {
    if (timer.current) clearTimeout(timer.current)
    timer.current = null
    if (draft === null) return
    setDraft(null)
    if (draft !== (committed ?? value)) {
      setCommitted(draft)
      onCommit(draft)
    }
  }

  const latestCommit = useLatest(commit)
  function commitAfterPause() {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(latestCommit, KEY_PAUSE_MS)
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
        onKeyUp={commitAfterPause}
        onBlur={commit}
      />
      <output className="slider-value">{format(shown)}</output>
    </label>
  )
}

/** A stable function that always calls the latest `fn`. */
function useLatest(fn: () => void): () => void {
  const ref = useRef(fn)
  useEffect(() => {
    ref.current = fn
  })
  return () => ref.current()
}
