import { useRef, useState } from 'react'

import { PHOTO_TYPES, PROJECT_TYPES, isSupportedFile } from '../api/documents'
import { useDropAnywhere } from '../hooks/useDropAnywhere'
import { droppedFiles, isMultiDrop } from '../lib/dropFiles'
import { ImportButtons } from './library/ImportButtons'

interface Props {
  onPick: (file: File) => void
  /** Bring many photos (or a folder) into the library. */
  onImport?: (files: File[]) => void
  busy?: boolean
  error?: string | null
}

/**
 * The start page's ways in: a new photo, a saved `.photoagent` project, or
 * many photos at once into the library. Any of them can also be dropped
 * anywhere on the page: one photo opens, several or a folder are imported.
 */
export function PhotoPicker({
  onPick,
  onImport,
  busy = false,
  error = null,
}: Props) {
  const photoInput = useRef<HTMLInputElement>(null)
  const projectInput = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState<string | null>(null)

  function pick(files: FileList | null | undefined) {
    const file = files?.[0]
    if (!file) return
    if (!isSupportedFile(file)) {
      setRejected(
        `${file.name} is not a JPEG, PNG, HEIC, TIFF, or RAW photo or a saved project.`,
      )
      return
    }
    setRejected(null)
    onPick(file)
  }

  useDropAnywhere(setDragging, (files, transfer) => {
    if (onImport && isMultiDrop(transfer)) {
      setRejected(null)
      void droppedFiles(transfer).then(onImport)
    } else pick(files)
  })

  const message = rejected ?? error
  return (
    <section className={`picker${dragging ? ' dragging' : ''}`}>
      <div className="picker-options">
        <div className="picker-option">
          <h2>New photo</h2>
          <p>Start editing a JPEG, PNG, HEIC, TIFF, or camera RAW photo.</p>
          <button
            type="button"
            className="primary"
            disabled={busy}
            onClick={() => photoInput.current?.click()}
          >
            {busy ? 'Opening…' : 'Choose a photo'}
          </button>
          <input
            ref={photoInput}
            type="file"
            accept={PHOTO_TYPES}
            hidden
            aria-label="Photo file"
            onChange={(e) => {
              pick(e.target.files)
              e.target.value = ''
            }}
          />
        </div>
        <div className="picker-option">
          <h2>Saved project</h2>
          <p>
            Pick up a <code>.photoagent</code> project with its layers and
            history.
          </p>
          <button
            type="button"
            disabled={busy}
            onClick={() => projectInput.current?.click()}
          >
            {busy ? 'Opening…' : 'Open a project'}
          </button>
          <input
            ref={projectInput}
            type="file"
            accept={PROJECT_TYPES}
            hidden
            aria-label="Project file"
            onChange={(e) => {
              pick(e.target.files)
              e.target.value = ''
            }}
          />
        </div>
        {onImport && (
          <div className="picker-option">
            <h2>Many photos</h2>
            <p>
              Import a folder or a whole shoot into your library to sort, cull,
              and edit together.
            </p>
            <div className="picker-buttons">
              <ImportButtons onImport={onImport} busy={busy} />
            </div>
          </div>
        )}
      </div>
      <p className="picker-hint">
        Or drop {onImport ? 'any of them' : 'either one'} anywhere on the page.
      </p>
      {message && (
        <p className="error" role="alert">
          {message}
        </p>
      )}
    </section>
  )
}
