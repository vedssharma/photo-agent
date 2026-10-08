import { useCallback, useEffect, useRef, useState } from 'react'

import { type DocumentView, type EditState, editByHand } from '../api/documents'

/** Computes the new edit state from the latest one. */
export type StateUpdate = (state: EditState) => EditState

export type EditFn = (
  label: string,
  update: StateUpdate,
  coalesce?: string,
) => Promise<void>

/**
 * Sends manual changes (layer settings, sliders, masks) to the server, one at a time, and
 * hands back the updated document. Each becomes a named step in the history.
 *
 * Changes are queued and each is applied to the state the previous one produced, so quick
 * successive tweaks never undo each other.
 */
export function useManualEdit(
  doc: DocumentView,
  onDocument: (doc: DocumentView) => void,
) {
  const [pending, setPending] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const queue = useRef(Promise.resolve())
  const latest = useRef(doc.state)
  const docId = doc.id
  useEffect(() => {
    latest.current = doc.state
  }, [doc.state])

  const edit: EditFn = useCallback(
    (label, update, coalesce) => {
      setPending((n) => n + 1)
      const run = async () => {
        setError(null)
        try {
          const state = update(latest.current)
          const next = await editByHand(docId, { label, state, coalesce })
          latest.current = next.state
          onDocument(next)
        } catch (err) {
          setError((err as Error).message)
        } finally {
          setPending((n) => n - 1)
        }
      }
      queue.current = queue.current.then(run)
      return queue.current
    },
    [docId, onDocument],
  )

  return { edit, working: pending > 0, error }
}
