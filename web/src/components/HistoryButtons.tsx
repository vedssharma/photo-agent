import { useEffect, useRef, useState } from 'react'

import { type DocumentView, redo, undo } from '../api/documents'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  /** True while the agent is editing; history is locked until it finishes. */
  disabled?: boolean
}

const isMac =
  typeof navigator !== 'undefined' && /Mac|iPhone|iPad/.test(navigator.platform)

function isTyping(target: EventTarget | null) {
  return (
    target instanceof HTMLElement &&
    (target.isContentEditable ||
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName))
  )
}

/** Undo and redo of agent turns, with the usual keyboard shortcuts. */
export function HistoryButtons({ doc, onDocument, disabled = false }: Props) {
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const canUndo = doc.can_undo && !disabled && !working
  const canRedo = doc.can_redo && !disabled && !working

  async function run(action: typeof undo) {
    setWorking(true)
    setError(null)
    try {
      onDocument(await action(doc.id))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  const latest = useRef({ canUndo, canRedo, run })
  useEffect(() => {
    latest.current = { canUndo, canRedo, run }
  })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!(isMac ? e.metaKey : e.ctrlKey) || isTyping(e.target)) return
      const key = e.key.toLowerCase()
      const wantsRedo = (key === 'z' && e.shiftKey) || key === 'y'
      const wantsUndo = key === 'z' && !e.shiftKey
      if (!wantsUndo && !wantsRedo) return
      e.preventDefault()
      const { canUndo, canRedo, run } = latest.current
      if (wantsUndo && canUndo) void run(undo)
      if (wantsRedo && canRedo) void run(redo)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const mod = isMac ? '⌘' : 'Ctrl+'
  const lastApplied = doc.turns.filter((t) => t.applied).at(-1)
  const nextRedo = doc.turns.find((t) => !t.applied)
  return (
    <div className="history">
      <button
        type="button"
        onClick={() => run(undo)}
        disabled={!canUndo}
        title={
          lastApplied
            ? `Undo “${lastApplied.request}” (${mod}Z)`
            : 'Nothing to undo'
        }
      >
        Undo
      </button>
      <button
        type="button"
        onClick={() => run(redo)}
        disabled={!canRedo}
        title={
          nextRedo
            ? `Redo “${nextRedo.request}” (${mod}Shift+Z)`
            : 'Nothing to redo'
        }
      >
        Redo
      </button>
      {error && (
        <span className="error" role="alert">
          {error}
        </span>
      )}
    </div>
  )
}
