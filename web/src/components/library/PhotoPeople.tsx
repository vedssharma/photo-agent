import { useEffect, useState } from 'react'

import { type Face, type Person, faceUrl, listFaces } from '../../api/library'
import { personName } from '../../lib/people'

interface Props {
  photoId: string
  /** Changes when the photo's faces may have changed. */
  version: number
  people: Person[]
  /** Say who a face is, or (null) that it is not who it was grouped with. */
  onChange: (face: Face, person: string | null) => void
}

/** Who is in one photo, with a way to correct it. */
export function PhotoPeople({ photoId, version, people, onChange }: Props) {
  const [found, setFound] = useState<{ photo: string; faces: Face[] } | null>(
    null,
  )

  useEffect(() => {
    let live = true
    listFaces(photoId).then(
      (faces) => live && setFound({ photo: photoId, faces }),
      () => live && setFound({ photo: photoId, faces: [] }),
    )
    return () => {
      live = false
    }
  }, [photoId, version])

  // While a change is being fetched, the faces as they were stay up.
  const faces = found?.photo === photoId ? found.faces : []
  if (faces.length === 0) return null
  return (
    <section className="info-people">
      <h3>People</h3>
      <ul className="faces">
        {faces.map((face) => {
          const who = face.person
            ? personName({ name: face.person_name })
            : 'Someone'
          return (
            <li key={face.id}>
              <img className="face" src={faceUrl(face.id)} alt="" />
              <span>{who}</span>
              {face.person && (
                <button
                  type="button"
                  className="link"
                  aria-label={`This is not ${who}`}
                  title="Not this person"
                  onClick={() => onChange(face, null)}
                >
                  ×
                </button>
              )}
              {people.length > 0 && (
                <select
                  aria-label={`Who is this? (${who})`}
                  value=""
                  onChange={(e) =>
                    e.target.value && onChange(face, e.target.value)
                  }
                >
                  <option value="">Is someone else…</option>
                  {people
                    .filter((p) => p.id !== face.person)
                    .map((p) => (
                      <option key={p.id} value={p.id}>
                        {personName(p)}
                      </option>
                    ))}
                </select>
              )}
            </li>
          )
        })}
      </ul>
    </section>
  )
}
