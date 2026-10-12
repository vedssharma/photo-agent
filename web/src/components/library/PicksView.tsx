import { type CullResult, thumbnailUrl } from '../../api/library'
import { Badges } from './Badges'

interface Props {
  result: CullResult
  onFlag: () => void
  onAlbum: () => void
  onEdit: (id: string) => void
  onClose: () => void
}

/** The best shots picked from a shoot, each with why. */
export function PicksView({ result, onFlag, onAlbum, onEdit, onClose }: Props) {
  return (
    <div className="picks">
      <div className="picks-head">
        <p role="status">
          {result.summary}
          {result.source === 'built-in' && result.picks.length > 0 && (
            <span className="muted"> (explained without Claude)</span>
          )}
        </p>
        <div className="picks-actions">
          <button
            type="button"
            className="primary"
            disabled={result.picks.length === 0}
            onClick={onFlag}
          >
            Flag as picks
          </button>
          <button
            type="button"
            disabled={result.picks.length === 0}
            onClick={onAlbum}
          >
            Make an album
          </button>
          <button type="button" className="link" onClick={onClose}>
            Back to photos
          </button>
        </div>
      </div>
      <ol className="picks-list" aria-label="Best shots">
        {result.picks.map((pick) => (
          <li key={pick.photo.id}>
            <button
              type="button"
              className="pick-thumb"
              aria-label={`Edit ${pick.photo.filename}`}
              onClick={() => onEdit(pick.photo.id)}
            >
              <img src={thumbnailUrl(pick.photo)} alt="" loading="lazy" />
              <Badges photo={pick.photo} />
            </button>
            <span className="pick-name">{pick.photo.filename}</span>
            <span className="pick-why">{pick.reasons.join(', ')}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}
