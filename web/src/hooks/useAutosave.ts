import { useEffect } from 'react'

import { type DocumentView, fetchGraph, fetchSource } from '../api/documents'
import type { ProjectStore } from '../lib/projectStore'

/** Wait this long after the last change before saving, so a burst of edits saves once. */
export const AUTOSAVE_DELAY_MS = 800

/**
 * Keep a copy of the open project (original photo plus edits) in the browser, updated
 * shortly after every change.
 */
export function useAutosave(
  doc: DocumentView,
  store: ProjectStore,
  /** Told about a failed save (a message) and about the next one that works (null). */
  onError?: (message: string | null) => void,
) {
  const { id, filename } = doc
  // Anything that changes what is shown or the history changes one of these.
  const version = `${doc.revision}:${doc.head}:${doc.history.length}:${doc.chat.length}`

  useEffect(() => {
    let cancelled = false
    const timer = setTimeout(async () => {
      try {
        const saved = await store.get(id)
        const original = saved?.original ?? (await fetchSource(id))
        const graph = await fetchGraph(id)
        if (cancelled) return
        await store.put({ id, filename, savedAt: Date.now(), graph, original })
        onError?.(null)
      } catch (err) {
        onError?.(
          `Could not save a copy in this browser: ${(err as Error).message}`,
        )
      }
    }, AUTOSAVE_DELAY_MS)
    return () => {
      cancelled = true
      clearTimeout(timer)
    }
  }, [id, filename, version, store, onError])
}
