import { LABELS, type PhotoFilter } from '../../api/library'

interface Props {
  filter: PhotoFilter
  onFilter: (filter: PhotoFilter) => void
}

/** Narrow the photos shown by rating, flag, and color label. */
export function FilterBar({ filter, onFilter }: Props) {
  const set = (change: PhotoFilter) => onFilter({ ...filter, ...change })
  return (
    <div className="filter-bar" role="group" aria-label="Filters">
      <select
        aria-label="Rating"
        value={filter.min_rating ?? 0}
        onChange={(e) => set({ min_rating: Number(e.target.value) })}
      >
        <option value={0}>Any rating</option>
        {[1, 2, 3, 4, 5].map((n) => (
          <option key={n} value={n}>
            {'★'.repeat(n)}
            {n < 5 ? ' or more' : ''}
          </option>
        ))}
      </select>
      <select
        aria-label="Flag"
        value={filter.flag ?? ''}
        onChange={(e) =>
          set({ flag: (e.target.value || null) as PhotoFilter['flag'] })
        }
      >
        <option value="">Any flag</option>
        <option value="picked">Picked</option>
        <option value="none">Unflagged</option>
        <option value="rejected">Rejected</option>
      </select>
      <select
        aria-label="Color label"
        value={filter.label ?? ''}
        onChange={(e) =>
          set({ label: (e.target.value || null) as PhotoFilter['label'] })
        }
      >
        <option value="">Any label</option>
        {LABELS.map((l) => (
          <option key={l} value={l}>
            {l[0].toUpperCase() + l.slice(1)}
          </option>
        ))}
      </select>
    </div>
  )
}
