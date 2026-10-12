import type { ReactNode } from 'react'

import { type LibraryPhoto, largeUrl } from '../../api/library'
import type { ClickKeys, Selection } from '../../lib/selection'
import { PhotoGrid } from './PhotoGrid'

interface Props {
  photos: LibraryPhoto[]
  selection: Selection
  onSelect: (id: string, keys: ClickKeys) => void
  onEdit: (id: string) => void
  badges?: (photo: LibraryPhoto) => ReactNode
}

/** One photo large, with the rest in a filmstrip underneath to flick through. */
export function Loupe({ photos, selection, onSelect, onEdit, badges }: Props) {
  const photo =
    photos.find((p) => p.id === selection.focus) ?? photos[0] ?? null
  return (
    <div className="loupe">
      <div className="loupe-stage">
        {photo ? (
          <img
            key={photo.id}
            src={largeUrl(photo)}
            alt={photo.filename}
            onDoubleClick={() => onEdit(photo.id)}
          />
        ) : (
          <p className="empty">No photo to show.</p>
        )}
      </div>
      <PhotoGrid
        variant="strip"
        label="Filmstrip"
        photos={photos}
        selection={selection}
        onSelect={onSelect}
        onOpen={onEdit}
        badges={badges}
      />
    </div>
  )
}
