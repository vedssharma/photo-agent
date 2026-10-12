import { type ReactNode, useEffect, useRef } from 'react'

import { type LibraryPhoto, thumbnailUrl } from '../../api/library'
import { type ClickKeys, type Selection, clickKeys } from '../../lib/selection'

interface Props {
  photos: LibraryPhoto[]
  selection: Selection
  onSelect: (id: string, keys: ClickKeys) => void
  /** Look at one photo large (double-click or Enter). */
  onOpen: (id: string) => void
  /** Extra marks drawn over a thumbnail (rating, flag, label, ...). */
  badges?: (photo: LibraryPhoto) => ReactNode
  /** Grid of thumbnails, or one row of them under the loupe. */
  variant?: 'grid' | 'strip'
  label?: string
}

/** Thumbnails of library photos, to select (click, shift-click, ctrl/cmd-click). */
export function PhotoGrid({
  photos,
  selection,
  onSelect,
  onOpen,
  badges,
  variant = 'grid',
  label = 'Photos',
}: Props) {
  const focused = useRef<HTMLLIElement>(null)
  useEffect(() => {
    focused.current?.scrollIntoView?.({ block: 'nearest', inline: 'nearest' })
  }, [selection.focus])

  return (
    <ul
      className={variant === 'grid' ? 'photo-grid' : 'filmstrip'}
      role="listbox"
      aria-label={label}
      aria-multiselectable="true"
    >
      {photos.map((photo) => {
        const selected = selection.ids.has(photo.id)
        const isFocus = selection.focus === photo.id
        return (
          <li
            key={photo.id}
            ref={isFocus ? focused : undefined}
            role="option"
            aria-selected={selected}
            aria-label={photo.filename}
            className={`thumb${selected ? ' selected' : ''}${isFocus ? ' focus' : ''}`}
            onClick={(e) => onSelect(photo.id, clickKeys(e))}
            onDoubleClick={() => onOpen(photo.id)}
            title={photo.filename}
          >
            <img
              src={thumbnailUrl(photo)}
              alt=""
              loading="lazy"
              draggable={false}
            />
            {badges?.(photo)}
          </li>
        )
      })}
    </ul>
  )
}
