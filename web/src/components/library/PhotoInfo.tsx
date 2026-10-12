import type { ReactNode } from 'react'

import type { Album, LibraryPhoto, Marks } from '../../api/library'
import { MarkControls } from './MarkControls'

interface Props {
  /** The selected photos, in display order. */
  photos: LibraryPhoto[]
  albums: Album[]
  onMark: (marks: Omit<Marks, 'ids'>) => void
  onAddToAlbum: (albumId: string) => void
  onRemoveFromAlbum: (albumId: string) => void
  /** More about a single photo (its details), below the marks. */
  children?: ReactNode
}

/** A value if every photo agrees on it, else null. */
function agreed<T>(
  photos: LibraryPhoto[],
  get: (p: LibraryPhoto) => T,
): T | null {
  if (photos.length === 0) return null
  const first = get(photos[0])
  return photos.every((p) => get(p) === first) ? first : null
}

/** Marks and albums for the selected photos. */
export function PhotoInfo({
  photos,
  albums,
  onMark,
  onAddToAlbum,
  onRemoveFromAlbum,
  children,
}: Props) {
  if (photos.length === 0) {
    return (
      <aside className="library-info" aria-label="Photo info">
        <p className="empty">Select a photo to rate, flag, or label it.</p>
      </aside>
    )
  }
  const one = photos.length === 1 ? photos[0] : null
  const names = new Map(albums.map((a) => [a.id, a.name]))
  const inAlbums = one ? one.albums.filter((id) => names.has(id)) : []
  return (
    <aside className="library-info" aria-label="Photo info">
      <h2 className="info-title">
        {one ? one.filename : `${photos.length} photos selected`}
      </h2>
      <MarkControls
        rating={agreed(photos, (p) => p.rating)}
        flag={agreed(photos, (p) => p.flag)}
        label={agreed(photos, (p) => p.label ?? null)}
        onMark={onMark}
      />
      <section className="info-albums">
        <h3>Albums</h3>
        {inAlbums.length > 0 && (
          <ul className="chips">
            {inAlbums.map((id) => (
              <li key={id}>
                {names.get(id)}
                <button
                  type="button"
                  className="link"
                  aria-label={`Take out of ${names.get(id)}`}
                  onClick={() => onRemoveFromAlbum(id)}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
        )}
        {albums.length > 0 ? (
          <select
            aria-label="Add to album"
            value=""
            onChange={(e) => e.target.value && onAddToAlbum(e.target.value)}
          >
            <option value="">Add to album…</option>
            {albums.map((a) => (
              <option key={a.id} value={a.id}>
                {a.name}
              </option>
            ))}
          </select>
        ) : (
          <p className="empty">Make an album from the sidebar.</p>
        )}
      </section>
      {children}
    </aside>
  )
}
