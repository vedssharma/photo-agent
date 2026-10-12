import { useRef } from 'react'

import { PHOTO_TYPES } from '../../api/documents'

interface Props {
  onImport: (files: File[]) => void
  busy?: boolean
}

/** Bring many photos into the library: pick several files, or a whole folder. */
export function ImportButtons({ onImport, busy = false }: Props) {
  const filesInput = useRef<HTMLInputElement>(null)
  const folderInput = useRef<HTMLInputElement>(null)
  const take = (input: HTMLInputElement) => {
    const files = Array.from(input.files ?? [])
    input.value = ''
    if (files.length) onImport(files)
  }
  return (
    <>
      <button
        type="button"
        disabled={busy}
        onClick={() => folderInput.current?.click()}
      >
        Import a folder
      </button>
      <input
        ref={folderInput}
        type="file"
        hidden
        multiple
        aria-label="Folder to import"
        // Lets the browser pick a folder and hand over every file inside it.
        {...{ webkitdirectory: '', directory: '' }}
        onChange={(e) => take(e.target)}
      />
      <button
        type="button"
        disabled={busy}
        onClick={() => filesInput.current?.click()}
      >
        Import photos
      </button>
      <input
        ref={filesInput}
        type="file"
        hidden
        multiple
        accept={PHOTO_TYPES}
        aria-label="Photos to import"
        onChange={(e) => take(e.target)}
      />
    </>
  )
}
