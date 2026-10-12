import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import {
  type Album,
  type Facets,
  type IndexStatus,
  type Label,
  type LibraryPhoto,
  type Marks,
  type PhotoFilter,
  type SearchResult,
  addToAlbum,
  changeTags,
  createAlbum,
  deleteAlbum,
  fetchFacets,
  filterQuery,
  libraryStatus,
  listAlbums,
  listPhotos,
  markPhotos,
  removeFromAlbum,
  removePhotos,
  renameAlbum,
  searchLibrary,
} from '../../api/library'
import { useDropAnywhere } from '../../hooks/useDropAnywhere'
import { type Importer, importText } from '../../hooks/useImporter'
import { droppedFiles } from '../../lib/dropFiles'
import {
  type ClickKeys,
  NOTHING,
  type Selection,
  clickSelect,
  keepOnly,
  moveFocus,
  selectAll,
  selectedIds,
} from '../../lib/selection'
import { Badges } from './Badges'
import { FilterBar } from './FilterBar'
import { ImportButtons } from './ImportButtons'
import { LibrarySidebar } from './LibrarySidebar'
import { Loupe } from './Loupe'
import { PhotoGrid } from './PhotoGrid'
import { PhotoDetails } from './PhotoDetails'
import { PhotoInfo } from './PhotoInfo'
import { SearchBar } from './SearchBar'
import { TagEditor } from './TagEditor'

/** Keys that mark the selection, as in other photo apps. */
const MARK_KEYS: Record<string, Omit<Marks, 'ids'>> = {
  '0': { rating: 0 },
  '1': { rating: 1 },
  '2': { rating: 2 },
  '3': { rating: 3 },
  '4': { rating: 4 },
  '5': { rating: 5 },
  p: { flag: 'picked' },
  x: { flag: 'rejected' },
  u: { flag: 'none' },
  ...Object.fromEntries(
    (['red', 'yellow', 'green', 'blue'] satisfies Label[]).map((label, i) => [
      String(6 + i),
      { label },
    ]),
  ),
}

const badges = (photo: LibraryPhoto) => <Badges photo={photo} />

type View = 'grid' | 'loupe'

interface Props {
  importer: Importer
  /** Open a photo in the editor. */
  onEdit: (id: string) => void
  /** How often to check on indexing while it runs, in ms. */
  pollMs?: number
}

/** Whether a key press belongs to a text field rather than the library. */
function typing(e: KeyboardEvent): boolean {
  const el = e.target as HTMLElement | null
  return (
    !!el &&
    (el.isContentEditable ||
      ['INPUT', 'TEXTAREA', 'SELECT'].includes(el.tagName))
  )
}

/** How many thumbnails fit in a row of the grid, for moving up and down. */
function columns(grid: Element | null): number {
  if (!grid) return 1
  const template = getComputedStyle(grid).gridTemplateColumns
  const count = template ? template.split(' ').filter(Boolean).length : 1
  return Math.max(1, count)
}

/** The whole collection: import, browse as a grid or one at a time, and act on a selection. */
export function LibraryView({ importer, onEdit, pollMs = 1500 }: Props) {
  const [photos, setPhotos] = useState<LibraryPhoto[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [picked, setSelection] = useState<Selection>(NOTHING)
  const [view, setView] = useState<View>('grid')
  const [status, setStatus] = useState<IndexStatus | null>(null)
  const [filter, setFilter] = useState<PhotoFilter>({})
  const [albums, setAlbums] = useState<Album[]>([])
  const [facets, setFacets] = useState<Facets | null>(null)
  const [search, setSearch] = useState<SearchResult | null>(null)
  const [searching, setSearching] = useState(false)
  const main = useRef<HTMLDivElement>(null)

  // Bumped to fetch the photo list again.
  const [version, setVersion] = useState(0)
  const reload = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    let live = true
    Promise.all([listPhotos(filter), listAlbums(), fetchFacets()]).then(
      ([found, albumList, facetList]) => {
        if (!live) return
        setPhotos(found)
        setAlbums(albumList)
        setFacets(facetList)
        setError(null)
      },
      (err: Error) => {
        if (live) setError(`Could not load the library: ${err.message}`)
      },
    )
    return () => {
      live = false
    }
  }, [version, importer.batches, filter])

  // Watch indexing while there is any, refreshing as photos finish.
  const lastPending = useRef<number | null>(null)
  useEffect(() => {
    let live = true
    let timer: ReturnType<typeof setTimeout> | undefined
    const check = async () => {
      try {
        const next = await libraryStatus()
        if (!live) return
        if (
          lastPending.current !== null &&
          lastPending.current !== next.pending
        )
          reload()
        lastPending.current = next.pending
        setStatus(next)
        if (next.pending > 0) timer = setTimeout(check, pollMs)
      } catch {
        // Indexing status is a nicety; the library works without it.
      }
    }
    void check()
    return () => {
      live = false
      clearTimeout(timer)
    }
  }, [importer.batches, pollMs, reload])

  const shown = useMemo(() => {
    if (!search) return photos ?? []
    // Keep the search's order, with each photo as it is now (marks change).
    const latest = new Map((photos ?? []).map((p) => [p.id, p]))
    return search.photos.map((p) => latest.get(p.id) ?? p)
  }, [photos, search])
  const order = useMemo(() => shown.map((p) => p.id), [shown])
  // Photos that left (deleted, or filtered out) drop out of the selection.
  const selection = keepOnly(picked, order)
  const chosen = selectedIds(selection, order)
  const focused =
    chosen.length === 1 ? (shown.find((p) => p.id === chosen[0]) ?? null) : null

  const select = useCallback(
    (id: string, keys: ClickKeys) =>
      setSelection((sel) => clickSelect(sel, order, id, keys)),
    [order],
  )

  const lookAt = (id: string) => {
    setSelection((sel) => clickSelect(sel, order, id))
    setView('loupe')
  }

  /** Run a change, then show the library as it is after it. */
  async function change(what: string, run: () => Promise<unknown>) {
    try {
      await run()
      reload()
    } catch (err) {
      setError(`Could not ${what}: ${(err as Error).message}`)
    }
  }

  const mark = (marks: Omit<Marks, 'ids'>) => {
    if (chosen.length)
      void change('mark the photos', () =>
        markPhotos({ ...marks, ids: chosen }),
      )
  }

  async function runSearch(query: string) {
    setSearching(true)
    try {
      setSearch(await searchLibrary(query, filter.album ?? null))
      setError(null)
    } catch (err) {
      setError(`Could not search: ${(err as Error).message}`)
    } finally {
      setSearching(false)
    }
  }

  const showFilter = (next: PhotoFilter) => {
    setSearch(null)
    setFilter(next)
  }

  function newAlbum() {
    const name = window.prompt(
      chosen.length
        ? `Name the new album for the ${chosen.length} selected photo${chosen.length === 1 ? '' : 's'}:`
        : 'Name the new album:',
    )
    if (name?.trim())
      void change('make the album', () => createAlbum(name.trim(), chosen))
  }

  function rename(album: Album) {
    const name = window.prompt('Rename the album:', album.name)
    if (name?.trim() && name.trim() !== album.name)
      void change('rename the album', () => renameAlbum(album.id, name.trim()))
  }

  function dropAlbum(album: Album) {
    if (
      !window.confirm(
        `Delete the album “${album.name}”? Its photos stay in the library.`,
      )
    )
      return
    setFilter({})
    void change('delete the album', () => deleteAlbum(album.id))
  }

  async function remove(ids: string[]) {
    if (ids.length === 0) return
    const what = ids.length === 1 ? 'this photo' : `these ${ids.length} photos`
    if (
      !window.confirm(`Delete ${what} and their edits? This cannot be undone.`)
    )
      return
    try {
      await removePhotos(ids)
      reload()
    } catch (err) {
      setError(`Could not delete: ${(err as Error).message}`)
    }
  }

  // Keyboard: arrows move, Enter looks closer, Escape steps back.
  const keys = useRef<(e: KeyboardEvent) => void>(() => {})
  const onKey = (e: KeyboardEvent) => {
    if (typing(e) || e.altKey) return
    const step =
      view === 'grid'
        ? columns(main.current?.querySelector('.photo-grid') ?? null)
        : 1
    const moves: Record<string, number> = {
      ArrowLeft: -1,
      ArrowRight: 1,
      ArrowUp: -step,
      ArrowDown: step,
    }
    if (e.key in moves && !e.metaKey && !e.ctrlKey) {
      e.preventDefault()
      setSelection((sel) => moveFocus(sel, order, moves[e.key], e.shiftKey))
    } else if (e.key === 'Enter' && selection.focus) {
      e.preventDefault()
      if (view === 'grid') setView('loupe')
      else onEdit(selection.focus)
    } else if (e.key === 'Escape' && view === 'loupe') {
      setView('grid')
    } else if (e.key === 'a' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      setSelection((sel) => selectAll(order, sel))
    } else if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault()
      void remove(chosen)
    } else if (e.key.toLowerCase() in MARK_KEYS && !e.metaKey && !e.ctrlKey) {
      e.preventDefault()
      mark(MARK_KEYS[e.key.toLowerCase()])
    }
  }
  useEffect(() => {
    keys.current = onKey
  })
  useEffect(() => {
    const listen = (e: KeyboardEvent) => keys.current(e)
    window.addEventListener('keydown', listen)
    return () => window.removeEventListener('keydown', listen)
  }, [])

  const [dragging, setDragging] = useState(false)
  useDropAnywhere(setDragging, (files, transfer) => {
    void droppedFiles(transfer).then((all) =>
      importer.start(all.length ? all : Array.from(files)),
    )
  })

  const importing = importText(importer.progress)
  const indexing =
    status && status.pending > 0
      ? `Indexing ${status.pending} of ${status.total} photos${status.current ? ` (${status.current})` : ''}…`
      : null

  return (
    <main className={`library${dragging ? ' dragging' : ''}`}>
      <LibrarySidebar
        albums={albums}
        filter={filter}
        onFilter={showFilter}
        onNewAlbum={newAlbum}
        onRenameAlbum={rename}
        onDeleteAlbum={dropAlbum}
      />
      <section className="library-main" ref={main}>
        <div className="library-toolbar">
          <ImportButtons
            onImport={(files) => void importer.start(files)}
            busy={importer.progress.running}
          />
          <SearchBar onSearch={(q) => void runSearch(q)} busy={searching} />
          <span className="library-count">
            {photos === null
              ? 'Loading…'
              : `${shown.length} photo${shown.length === 1 ? '' : 's'}`}
            {chosen.length > 1 && `, ${chosen.length} selected`}
          </span>
          <span className="spacer" />
          <div className="segmented" role="group" aria-label="View">
            <button
              type="button"
              aria-pressed={view === 'grid'}
              onClick={() => setView('grid')}
            >
              Grid
            </button>
            <button
              type="button"
              aria-pressed={view === 'loupe'}
              onClick={() => setView('loupe')}
              disabled={shown.length === 0}
            >
              Filmstrip
            </button>
          </div>
          <button
            type="button"
            className="primary"
            disabled={!selection.focus}
            onClick={() => selection.focus && onEdit(selection.focus)}
          >
            Edit
          </button>
          <button
            type="button"
            disabled={chosen.length === 0}
            onClick={() => void remove(chosen)}
            title="Delete the selected photos and their edits"
          >
            Delete
          </button>
        </div>
        {search ? (
          <p className="search-summary" role="status">
            {search.photos.length} photo{search.photos.length === 1 ? '' : 's'}:{' '}
            {search.summary}
            {search.source === 'built-in' && ' (read without Claude)'}
            <button
              type="button"
              className="link"
              onClick={() => setSearch(null)}
            >
              Clear search
            </button>
          </p>
        ) : (
          <FilterBar filter={filter} onFilter={setFilter} facets={facets} />
        )}
        {(importing || indexing || error) && (
          <p className="library-status" role="status">
            {error && <span className="error">{error} </span>}
            {importing}
            {importing && indexing && ' '}
            {indexing}
            {importer.progress.total > 0 && !importer.progress.running && (
              <button type="button" className="link" onClick={importer.dismiss}>
                Dismiss
              </button>
            )}
          </p>
        )}
        {photos !== null && shown.length === 0 ? (
          search || Object.keys(filterQuery(filter)).length > 0 ? (
            <div className="library-empty">
              <h2>No photos here</h2>
              <p>
                <button
                  type="button"
                  className="link"
                  onClick={() => showFilter({})}
                >
                  Show all photos
                </button>
              </p>
            </div>
          ) : (
            <div className="library-empty">
              <h2>Your library is empty</h2>
              <p>
                Import a folder or a handful of photos, or drop them anywhere
                here.
              </p>
            </div>
          )
        ) : view === 'grid' ? (
          <PhotoGrid
            photos={shown}
            selection={selection}
            onSelect={select}
            onOpen={lookAt}
            badges={badges}
          />
        ) : (
          <Loupe
            photos={shown}
            selection={selection}
            onSelect={select}
            onEdit={onEdit}
            badges={badges}
          />
        )}
      </section>
      <PhotoInfo
        photos={shown.filter((p) => selection.ids.has(p.id))}
        albums={albums}
        onMark={mark}
        onAddToAlbum={(id) =>
          void change('add to the album', () => addToAlbum(id, chosen))
        }
        onRemoveFromAlbum={(id) =>
          void change('take out of the album', () =>
            removeFromAlbum(id, chosen),
          )
        }
      >
        <TagEditor
          tags={focused?.tags ?? []}
          onAdd={(tag) =>
            void change('add the tag', () => changeTags(chosen, { add: [tag] }))
          }
          onRemove={(tag) =>
            void change('remove the tag', () =>
              changeTags(chosen, { remove: [tag] }),
            )
          }
          onShow={(tag) => showFilter({ tags: [tag] })}
        />
        {focused && <PhotoDetails photo={focused} />}
      </PhotoInfo>
    </main>
  )
}
