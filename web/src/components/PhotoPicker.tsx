import { useRef, useState } from 'react'

import { ACCEPTED_TYPES, isSupportedFile } from '../api/documents'
import { useDropAnywhere } from '../hooks/useDropAnywhere'

interface Props {
  onPick: (file: File) => void
  busy?: boolean
  error?: string | null
}

/**
 * Lets the user choose a photo with a file dialog or by dropping it anywhere on the page.
 */
export function PhotoPicker({ onPick, busy = false, error = null }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [rejected, setRejected] = useState<string | null>(null)

  function pick(files: FileList | null | undefined) {
    const file = files?.[0]
    if (!file) return
    if (!isSupportedFile(file)) {
      setRejected(`${file.name} is not a JPEG, PNG, or HEIC photo.`)
      return
    }
    setRejected(null)
    onPick(file)
  }

  useDropAnywhere(setDragging, (files) => pick(files))

  const message = rejected ?? error
  return (
    <section className={`picker${dragging ? ' dragging' : ''}`}>
      <h2>Open a photo</h2>
      <p>
        Drop a JPEG, PNG, or HEIC photo anywhere on the page, or choose one.
      </p>
      <button
        type="button"
        className="primary"
        disabled={busy}
        onClick={() => input.current?.click()}
      >
        {busy ? 'Opening…' : 'Choose a photo'}
      </button>
      <input
        ref={input}
        type="file"
        accept={ACCEPTED_TYPES}
        hidden
        aria-label="Photo file"
        onChange={(e) => {
          pick(e.target.files)
          e.target.value = ''
        }}
      />
      {message && (
        <p className="error" role="alert">
          {message}
        </p>
      )}
    </section>
  )
}
