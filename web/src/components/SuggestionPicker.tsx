import { useEffect, useState } from 'react'

import {
  type SuggestionSet,
  applySuggestion,
  fetchSuggestions,
  suggestionPreviewUrl,
} from '../api/advice'
import type { DocumentView } from '../api/documents'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  disabled?: boolean
}

/** Thumbnails of a few directions the photo could go in; picking one applies it. */
export function SuggestionPicker({ doc, onDocument, disabled = false }: Props) {
  const [found, setFound] = useState<SuggestionSet | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [working, setWorking] = useState(false)

  useEffect(() => {
    let live = true
    fetchSuggestions(doc.id)
      .then((set) => {
        if (live) setFound(set)
      })
      .catch((err: Error) => {
        if (live) setError(err.message)
      })
    return () => {
      live = false
    }
  }, [doc.id])

  async function pick(id: string) {
    setWorking(true)
    setError(null)
    try {
      onDocument(await applySuggestion(doc.id, id))
    } catch (err) {
      setError(`Could not apply it: ${(err as Error).message}`)
      setWorking(false)
    }
  }

  if (error)
    return (
      <p className="fine-print" role="status">
        No suggestions right now ({error}).
      </p>
    )
  if (!found)
    return (
      <p className="thinking" role="status">
        Looking for ideas for this photo…
      </p>
    )
  return (
    <div className="suggestion-picker">
      <p>A few directions to start from:</p>
      <ul>
        {found.suggestions.map((s) => (
          <li key={s.id}>
            <button
              type="button"
              disabled={disabled || working}
              onClick={() => void pick(s.id)}
              title={s.description}
            >
              <img src={suggestionPreviewUrl(doc, s)} alt="" loading="lazy" />
              <strong>{s.title}</strong>
              <span>{s.description}</span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  )
}
