import { useCallback, useRef, useState } from 'react'

import {
  type Skipped,
  importBatches,
  importPhotos,
  photoFiles,
} from '../api/library'

export interface ImportProgress {
  /** Photos in this import. */
  total: number
  /** Photos sent so far (imported or skipped). */
  done: number
  imported: number
  skipped: Skipped[]
  running: boolean
  error: string | null
}

export const IDLE: ImportProgress = {
  total: 0,
  done: 0,
  imported: 0,
  skipped: [],
  running: false,
  error: null,
}

/**
 * Imports many photos into the library a batch at a time, reporting progress.
 * `batches` counts finished batches, so the library can refresh as photos land.
 */
export function useImporter() {
  const [progress, setProgress] = useState<ImportProgress>(IDLE)
  const [batches, setBatches] = useState(0)
  const running = useRef(false)

  const start = useCallback(async (files: File[]) => {
    if (running.current) return
    const photos = photoFiles(files)
    const ignored = files.length - photos.length
    if (photos.length === 0) {
      setProgress({
        ...IDLE,
        error: ignored
          ? 'None of those files are JPEG, PNG, or HEIC photos.'
          : 'There were no files to import.',
      })
      return
    }
    running.current = true
    let state: ImportProgress = { ...IDLE, total: photos.length, running: true }
    setProgress(state)
    try {
      for (const batch of importBatches(photos)) {
        const result = await importPhotos(batch)
        state = {
          ...state,
          done: state.done + batch.length,
          imported: state.imported + result.imported.length,
          skipped: [...state.skipped, ...result.skipped],
        }
        setProgress(state)
        setBatches((n) => n + 1)
      }
      state = { ...state, running: false }
    } catch (err) {
      state = {
        ...state,
        running: false,
        error: `The import stopped: ${(err as Error).message}`,
      }
    } finally {
      running.current = false
      setProgress(state)
    }
  }, [])

  const dismiss = useCallback(() => setProgress(IDLE), [])

  return { progress, batches, start, dismiss }
}

export type Importer = ReturnType<typeof useImporter>

/** One line about an import, for the library's status bar. */
export function importText(p: ImportProgress): string | null {
  if (p.error) return p.error
  if (p.running) return `Importing ${p.done} of ${p.total} photos…`
  if (p.total === 0) return null
  const parts = [`Imported ${p.imported} of ${p.total} photos.`]
  const copies = p.skipped.filter((s) => s.existing).length
  if (copies) parts.push(`${copies} already in the library.`)
  const failed = p.skipped.length - copies
  if (failed) parts.push(`${failed} could not be read.`)
  return parts.join(' ')
}
