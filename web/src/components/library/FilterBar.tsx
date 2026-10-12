import { type Facets, LABELS, type PhotoFilter } from '../../api/library'

interface Props {
  filter: PhotoFilter
  onFilter: (filter: PhotoFilter) => void
  facets: Facets | null
}

/** The year a date range covers exactly, if it is one whole year. */
function wholeYear(filter: PhotoFilter): string {
  const from = filter.taken_after ?? ''
  const to = filter.taken_before ?? ''
  const year = from.slice(0, 4)
  return from === `${year}-01-01` && to === `${year}-12-31` ? year : ''
}

/** Narrow the photos shown by rating, flag, label, date, camera, and place. */
export function FilterBar({ filter, onFilter, facets }: Props) {
  const set = (change: PhotoFilter) => onFilter({ ...filter, ...change })
  const year = wholeYear(filter)
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
      {facets && facets.years.length > 0 && (
        <select
          aria-label="Year"
          value={year}
          onChange={(e) =>
            set(
              e.target.value
                ? {
                    taken_after: `${e.target.value}-01-01`,
                    taken_before: `${e.target.value}-12-31`,
                  }
                : { taken_after: null, taken_before: null },
            )
          }
        >
          <option value="">Any year</option>
          {facets.years.map((y) => (
            <option key={y.value} value={y.value}>
              {y.label} ({y.count})
            </option>
          ))}
        </select>
      )}
      <label className="date-field">
        From
        <input
          type="date"
          value={filter.taken_after ?? ''}
          onChange={(e) => set({ taken_after: e.target.value || null })}
        />
      </label>
      <label className="date-field">
        To
        <input
          type="date"
          value={filter.taken_before ?? ''}
          onChange={(e) => set({ taken_before: e.target.value || null })}
        />
      </label>
      {facets && facets.cameras.length > 0 && (
        <select
          aria-label="Camera"
          value={filter.camera ?? ''}
          onChange={(e) => set({ camera: e.target.value || null })}
        >
          <option value="">Any camera</option>
          {facets.cameras.map((c) => (
            <option key={c.value} value={c.value}>
              {c.label} ({c.count})
            </option>
          ))}
        </select>
      )}
      {facets && facets.places.length > 0 && (
        <select
          aria-label="Place"
          value={filter.place ?? ''}
          onChange={(e) => set({ place: e.target.value || null })}
        >
          <option value="">Anywhere</option>
          {facets.places.map((p) => (
            <option key={p.value} value={p.value}>
              Near {p.label} ({p.count})
            </option>
          ))}
        </select>
      )}
      <select
        aria-label="Sort"
        value={filter.sort ?? 'taken'}
        onChange={(e) =>
          set({ sort: e.target.value as NonNullable<PhotoFilter['sort']> })
        }
      >
        <option value="taken">Newest taken first</option>
        <option value="imported">Newest imported first</option>
      </select>
    </div>
  )
}
