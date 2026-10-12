import { useEffect, useState } from 'react'

import {
  type DuplicateGroup,
  dismissDuplicates,
  listDuplicates,
  removePhotos,
  thumbnailUrl,
} from '../../api/library'
import { Badges } from './Badges'

interface Props {
  /** Changes when the library has changed. */
  version: number
  /** Something was deleted or dismissed: show the library as it is now. */
  onChanged: () => void
  onError: (message: string) => void
}

const plural = (n: number, one: string, many = `${one}s`) =>
  `${n} ${n === 1 ? one : many}`

/** Review groups of duplicate photos and keep one of each. */
export function DuplicatesView({ version, onChanged, onError }: Props) {
  const [groups, setGroups] = useState<DuplicateGroup[] | null>(null)
  // The photo to keep in each group, where someone chose other than suggested.
  const [chosen, setChosen] = useState<Record<string, string>>({})

  useEffect(() => {
    let live = true
    listDuplicates().then(
      (found) => live && setGroups(found),
      (err: Error) =>
        live && onError(`Could not find duplicates: ${err.message}`),
    )
    return () => {
      live = false
    }
  }, [version, onError])

  if (groups === null) return <p className="library-empty">Looking…</p>
  if (groups.length === 0)
    return (
      <div className="library-empty">
        <h2>No duplicates</h2>
        <p>Photos still being indexed are compared once they are done.</p>
      </div>
    )

  const keeper = (group: DuplicateGroup) => chosen[group.keep] ?? group.keep
  const extras = (group: DuplicateGroup) =>
    group.photos.map((p) => p.id).filter((id) => id !== keeper(group))

  async function act(what: string, run: () => Promise<unknown>) {
    try {
      await run()
      onChanged()
    } catch (err) {
      onError(`Could not ${what}: ${(err as Error).message}`)
    }
  }

  function deleteExtras(ids: string[]) {
    if (
      window.confirm(
        `Delete ${plural(ids.length, 'duplicate photo')} and their edits? This cannot be undone.`,
      )
    )
      void act('delete the duplicates', () => removePhotos(ids))
  }

  const allExtras = groups.flatMap(extras)
  return (
    <div className="duplicates">
      <div className="duplicates-head">
        <p>
          {plural(groups.length, 'group')} of duplicates. The photo marked Keep
          is the one picked, rated, or edited, else the largest; click another
          to keep it instead.
        </p>
        <button type="button" onClick={() => deleteExtras(allExtras)}>
          Keep one of each, delete {plural(allExtras.length, 'other')}
        </button>
      </div>
      {groups.map((group) => (
        <section
          key={group.keep}
          className="duplicate-group"
          aria-label={`${group.kind === 'exact' ? 'Same photo' : 'Same shot'}: ${group.photos.map((p) => p.filename).join(', ')}`}
        >
          <h3>
            {group.kind === 'exact' ? 'The same photo' : 'The same shot'},{' '}
            {group.photos.length} times
          </h3>
          <ul>
            {group.photos.map((photo) => {
              const keep = photo.id === keeper(group)
              return (
                <li key={photo.id} className={keep ? 'keep' : undefined}>
                  <button
                    type="button"
                    aria-pressed={keep}
                    aria-label={`Keep ${photo.filename}`}
                    onClick={() =>
                      setChosen((c) => ({ ...c, [group.keep]: photo.id }))
                    }
                  >
                    <img src={thumbnailUrl(photo)} alt="" loading="lazy" />
                    {keep && <span className="keep-badge">Keep</span>}
                    <Badges photo={photo} />
                  </button>
                  <span className="duplicate-name">{photo.filename}</span>
                  <span className="duplicate-size">
                    {photo.width}×{photo.height}
                  </span>
                </li>
              )
            })}
          </ul>
          <div className="duplicate-actions">
            <button type="button" onClick={() => deleteExtras(extras(group))}>
              Delete the other {plural(extras(group).length, 'photo')}
            </button>
            <button
              type="button"
              className="link"
              onClick={() =>
                void act('mark them as different', () =>
                  dismissDuplicates(group.photos.map((p) => p.id)),
                )
              }
            >
              Not duplicates
            </button>
          </div>
        </section>
      ))}
    </div>
  )
}
