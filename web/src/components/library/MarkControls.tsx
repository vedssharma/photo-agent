import { type Flag, type Label, LABELS, type Marks } from '../../api/library'

interface Props {
  /** Current marks when one photo (or several that agree) is shown, else null. */
  rating: number | null
  flag: Flag | null
  label: Label | null | undefined
  onMark: (marks: Omit<Marks, 'ids'>) => void
  disabled?: boolean
}

/** Stars, pick/reject, and color label for the photos in hand. */
export function MarkControls({ rating, flag, label, onMark, disabled }: Props) {
  return (
    <div className="marks">
      <div className="stars" role="group" aria-label="Rating">
        {[1, 2, 3, 4, 5].map((n) => (
          <button
            key={n}
            type="button"
            className={`star${rating !== null && n <= rating ? ' on' : ''}`}
            aria-label={`${n} star${n === 1 ? '' : 's'}`}
            aria-pressed={rating === n}
            disabled={disabled}
            // Clicking the current rating again clears it, as in most photo apps.
            onClick={() => onMark({ rating: rating === n ? 0 : n })}
          >
            ★
          </button>
        ))}
      </div>
      <div className="segmented" role="group" aria-label="Flag">
        <button
          type="button"
          aria-pressed={flag === 'picked'}
          disabled={disabled}
          onClick={() =>
            onMark({ flag: flag === 'picked' ? 'none' : 'picked' })
          }
          title="Pick (P)"
        >
          Pick
        </button>
        <button
          type="button"
          aria-pressed={flag === 'rejected'}
          disabled={disabled}
          onClick={() =>
            onMark({ flag: flag === 'rejected' ? 'none' : 'rejected' })
          }
          title="Reject (X)"
        >
          Reject
        </button>
      </div>
      <div className="labels" role="group" aria-label="Color label">
        {LABELS.map((l) => (
          <button
            key={l}
            type="button"
            className={`label-dot ${l}${label === l ? ' on' : ''}`}
            aria-label={`${l[0].toUpperCase()}${l.slice(1)} label`}
            aria-pressed={label === l}
            disabled={disabled}
            onClick={() => onMark({ label: label === l ? 'none' : l })}
          />
        ))}
      </div>
    </div>
  )
}
