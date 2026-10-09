import { useEffect, useState } from 'react'

import type { DocumentView } from '../api/documents'
import {
  type StyleSummary,
  applyUsualLook,
  fetchStyle,
  forgetStyle,
} from '../api/style'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  disabled?: boolean
}

/** The person's taste as learned from their edits, and "my usual look". */
export function StylePanel({ doc, onDocument, disabled = false }: Props) {
  const [summary, setSummary] = useState<StyleSummary | null>(null)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Every edit can teach it something, so look again after each one.
  useEffect(() => {
    let live = true
    fetchStyle()
      .then((found) => {
        if (live) setSummary(found)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [doc.revision])

  async function run(task: () => Promise<void>) {
    setWorking(true)
    setError(null)
    try {
      await task()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  if (!summary) return null
  const locked = disabled || working
  return (
    <section className="panel style-panel" aria-label="Your style">
      <div className="panel-head">
        <h2>Your style</h2>
        <button
          type="button"
          className="icon"
          disabled={locked || !summary.has_usual_look}
          title={
            summary.has_usual_look
              ? 'Add one layer with the settings you keep coming back to'
              : 'Keep editing and downloading photos; it learns what you like'
          }
          onClick={() =>
            void run(async () => onDocument(await applyUsualLook(doc.id)))
          }
        >
          My usual look
        </button>
      </div>
      {summary.lines.length === 0 ? (
        <p className="empty">
          Nothing learned yet. Your downloads, hand tweaks, and undos teach it
          what you like.
        </p>
      ) : (
        <>
          <ul className="style-lines">
            {summary.lines.map((line) => (
              <li key={line}>You like {line}</li>
            ))}
          </ul>
          <button
            type="button"
            className="link"
            disabled={locked}
            onClick={() =>
              void run(async () => {
                await forgetStyle()
                setSummary(await fetchStyle())
              })
            }
          >
            Forget my style
          </button>
        </>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
