import { type Person, type PhotoFilter, faceUrl } from '../../api/library'
import { personName } from '../../lib/people'

interface Props {
  /** Whether people albums are on; null while loading. */
  on: boolean | null
  people: Person[]
  filter: PhotoFilter
  onFilter: (filter: PhotoFilter) => void
  onTurnOn: () => void
  onTurnOff: () => void
  onRename: (person: Person) => void
  onMerge: (person: Person, into: string) => void
  onForget: (person: Person) => void
}

/** People albums in the sidebar: off until turned on, then everyone found. */
export function PeopleSection({
  on,
  people,
  filter,
  onFilter,
  onTurnOn,
  onTurnOff,
  onRename,
  onMerge,
  onForget,
}: Props) {
  if (on === null) return null
  return (
    <section aria-label="People">
      <div className="side-head">
        <h2>People</h2>
        {on && (
          <button
            type="button"
            className="link"
            onClick={onTurnOff}
            title="Stop grouping photos by face and forget every face found"
          >
            Turn off
          </button>
        )}
      </div>
      {!on ? (
        <div className="people-off">
          <p className="empty">
            Group photos by who is in them. Faces are found and compared on this
            computer only, and never sent anywhere.
          </p>
          <button type="button" onClick={onTurnOn}>
            Find people
          </button>
        </div>
      ) : people.length === 0 ? (
        <p className="empty">
          Nobody yet. People show up once they are in two photos.
        </p>
      ) : (
        <ul className="sources people">
          {people.map((person) => {
            const open =
              filter.people?.length === 1 && filter.people[0] === person.id
            const others = people.filter((p) => p.id !== person.id)
            return (
              <li key={person.id} className={open ? 'open' : undefined}>
                <button
                  type="button"
                  aria-pressed={open}
                  onClick={() => onFilter({ people: [person.id] })}
                >
                  <img className="face" src={faceUrl(person.cover)} alt="" />
                  <span className="source-name">{personName(person)}</span>
                  <span className="source-count">{person.photos}</span>
                </button>
                {open && (
                  <span className="source-actions">
                    <button
                      type="button"
                      className="link"
                      onClick={() => onRename(person)}
                    >
                      {person.name ? 'Rename' : 'Name'}
                    </button>
                    {others.length > 0 && (
                      <select
                        aria-label={`${personName(person)} is the same person as`}
                        value=""
                        onChange={(e) =>
                          e.target.value && onMerge(person, e.target.value)
                        }
                      >
                        <option value="">Same person as…</option>
                        {others.map((p) => (
                          <option key={p.id} value={p.id}>
                            {personName(p)} ({p.photos})
                          </option>
                        ))}
                      </select>
                    )}
                    <button
                      type="button"
                      className="link"
                      onClick={() => onForget(person)}
                    >
                      Forget
                    </button>
                  </span>
                )}
              </li>
            )
          })}
        </ul>
      )}
    </section>
  )
}
