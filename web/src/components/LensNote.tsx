import { useEffect, useState } from 'react'

import { type LensInfo, fetchLens } from '../api/documents'

interface Props {
  docId: string
  /** A lens named by hand instead of the photo's own. */
  lensName?: string
  /** Overridable for tests. */
  load?: (docId: string, lensName: string) => Promise<LensInfo>
}

/** Says which measured lens profile a lens profile correction is using, if any. */
export function LensNote({ docId, lensName = '', load = fetchLens }: Props) {
  const [info, setInfo] = useState<LensInfo | null>(null)
  useEffect(() => {
    let current = true
    load(docId, lensName)
      .then((found) => current && setInfo(found))
      .catch(() => current && setInfo(null))
    return () => {
      current = false
    }
  }, [docId, lensName, load])

  if (!info) return null
  if (info.profile) {
    const at = info.focal_length ? ` at ${info.focal_length} mm` : ''
    return (
      <p className="hint">
        Using the profile for {info.profile}
        {at}.
      </p>
    )
  }
  return (
    <p className="hint">
      {info.lens
        ? `There is no profile for ${info.lens}.`
        : 'The photo does not say which lens took it.'}{' '}
      Name the lens below, or use Lens correction to fix it by hand.
    </p>
  )
}
