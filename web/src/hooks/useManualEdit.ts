import { useCallback, useRef, useState } from 'react'

import { type DocumentView, type EditState, editByHand } from '../api/documents'

export type EditFn = (
  label: string,
  state: EditState,
  coalesce?: string,
) => Promise<void>

/**
 * Sends manual changes (layer settings, sliders, masks) to the server, one at a time, and
 * hands back the updated document. Each becomes a named step in the history.
 */
export function useManualEdit(
  doc: DocumentView,
  onDocument: (doc: DocumentView) => void,
) {
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const queue = useRef(Promise.resolve())
  const docId = doc.id

  const edit: EditFn = useCallback(
    (label, state, coalesce) => {
      const run = async () => {
        setWorking(true)
        setError(null)
        try {
          onDocument(await editByHand(docId, { label, state, coalesce }))
        } catch (err) {
          setError((err as Error).message)
        } finally {
          setWorking(false)
        }
      }
      queue.current = queue.current.then(run)
      return queue.current
    },
    [docId, onDocument],
  )

  return { edit, working, error }
}
