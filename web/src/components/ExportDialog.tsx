import { useEffect, useState } from 'react'

import {
  type DocumentView,
  type ExportOptions,
  type ExportedFile,
  exportDocument,
  saveFile,
} from '../api/documents'

interface Props {
  doc: DocumentView
  onClose: () => void
  /** Overridable for tests; defaults to triggering a browser download. */
  save?: (file: ExportedFile) => void
}

/** Choose a format and quality, then render at full resolution and download. */
export function ExportDialog({ doc, onClose, save = saveFile }: Props) {
  const cutout = doc.state.cutout
  const transparent = !!cutout && cutout.visible && !cutout.background
  const [format, setFormat] = useState<ExportOptions['format']>(
    transparent ? 'png' : 'jpeg',
  )
  const [quality, setQuality] = useState(92)
  const [keepLocation, setKeepLocation] = useState(false)
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function download() {
    setWorking(true)
    setError(null)
    try {
      save(
        await exportDocument(doc, {
          format,
          quality,
          keep_location: keepLocation,
        }),
      )
      onClose()
    } catch (err) {
      setError(`Export failed: ${(err as Error).message}`)
    } finally {
      setWorking(false)
    }
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="export-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="export-title">Download photo</h2>
        <p className="status">
          Saved at full resolution ({doc.width} × {doc.height} before any crop).
        </p>

        <fieldset>
          <legend>Format</legend>
          <label>
            <input
              type="radio"
              name="format"
              checked={format === 'jpeg'}
              onChange={() => setFormat('jpeg')}
            />
            JPEG <span className="status">smaller, best for sharing</span>
          </label>
          <label>
            <input
              type="radio"
              name="format"
              checked={format === 'png'}
              onChange={() => setFormat('png')}
            />
            PNG{' '}
            <span className="status">
              {transparent
                ? 'keeps the transparent background'
                : 'lossless, larger file'}
            </span>
          </label>
        </fieldset>
        {transparent && format === 'jpeg' && (
          <p className="hint">
            JPEG cannot be transparent, so the background will be white.
          </p>
        )}

        {format === 'jpeg' && (
          <label className="quality">
            Quality {quality}
            <input
              type="range"
              min={50}
              max={100}
              value={quality}
              onChange={(e) => setQuality(Number(e.target.value))}
            />
          </label>
        )}

        <label>
          <input
            type="checkbox"
            checked={keepLocation}
            onChange={(e) => setKeepLocation(e.target.checked)}
          />
          Keep GPS location{' '}
          <span className="status">(removed by default for privacy)</span>
        </label>

        {error && (
          <p className="error" role="alert">
            {error}
          </p>
        )}

        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button
            type="button"
            className="primary"
            onClick={download}
            disabled={working}
          >
            {working ? 'Rendering…' : 'Download'}
          </button>
        </div>
      </div>
    </div>
  )
}
