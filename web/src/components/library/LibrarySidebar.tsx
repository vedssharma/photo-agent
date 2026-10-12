import type { ReactNode } from 'react'

import type { Album, PhotoFilter } from '../../api/library'

interface Props {
  albums: Album[]
  /** The filter shown, or null when something else (duplicates) is. */
  filter: PhotoFilter | null
  onFilter: (filter: PhotoFilter) => void
  /** Whether duplicates are being reviewed. */
  duplicates: boolean
  onDuplicates: () => void
  onNewAlbum: () => void
  onRenameAlbum: (album: Album) => void
  onDeleteAlbum: (album: Album) => void
  /** More sections (people, duplicates, ...) below the albums. */
  children?: ReactNode
}

/** Where to look: the whole library, the picks, or an album. */
export function LibrarySidebar({
  albums,
  filter: shown,
  onFilter,
  duplicates,
  onDuplicates,
  onNewAlbum,
  onRenameAlbum,
  onDeleteAlbum,
  children,
}: Props) {
  // Nothing in the photo list is pressed while duplicates are shown.
  const filter: PhotoFilter = shown ?? {}
  const everything =
    shown !== null && !filter.album && !filter.flag && !filter.people?.length
  return (
    <aside className="library-side" aria-label="Library sections">
      <section>
        <ul className="sources">
          <li>
            <button
              type="button"
              aria-pressed={everything}
              onClick={() => onFilter({})}
            >
              All photos
            </button>
          </li>
          <li>
            <button
              type="button"
              aria-pressed={
                shown !== null &&
                !filter.album &&
                !filter.people?.length &&
                filter.flag === 'picked'
              }
              onClick={() => onFilter({ flag: 'picked' })}
            >
              Picks
            </button>
          </li>
          <li>
            <button
              type="button"
              aria-pressed={duplicates}
              onClick={onDuplicates}
              title="Find copies of the same photo or shot, and keep one"
            >
              Duplicates
            </button>
          </li>
        </ul>
      </section>
      <section>
        <div className="side-head">
          <h2>Albums</h2>
          <button
            type="button"
            className="link"
            onClick={onNewAlbum}
            title="Make an album (with the selected photos in it)"
          >
            New
          </button>
        </div>
        {albums.length === 0 ? (
          <p className="empty">No albums yet.</p>
        ) : (
          <ul className="sources">
            {albums.map((album) => {
              const open = filter.album === album.id
              return (
                <li key={album.id} className={open ? 'open' : undefined}>
                  <button
                    type="button"
                    aria-pressed={open}
                    onClick={() => onFilter({ album: album.id })}
                  >
                    <span className="source-name">{album.name}</span>
                    <span className="source-count">{album.count}</span>
                  </button>
                  {open && (
                    <span className="source-actions">
                      <button
                        type="button"
                        className="link"
                        onClick={() => onRenameAlbum(album)}
                      >
                        Rename
                      </button>
                      <button
                        type="button"
                        className="link"
                        onClick={() => onDeleteAlbum(album)}
                      >
                        Delete album
                      </button>
                    </span>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </section>
      {children}
    </aside>
  )
}
