import type { ProjectSummary } from '../lib/projectStore'

interface Props {
  projects: ProjectSummary[]
  busy?: boolean
  onOpen: (id: string) => void
}

const when = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

/** Projects saved in this browser, newest first. */
export function RecentProjects({ projects, busy = false, onOpen }: Props) {
  if (projects.length === 0) return null
  return (
    <section className="recent" aria-label="Recent projects">
      <h2>Continue editing</h2>
      <ul>
        {projects.map((p) => (
          <li key={p.id}>
            <button type="button" disabled={busy} onClick={() => onOpen(p.id)}>
              <span className="recent-name">{p.filename}</span>
              <span className="recent-when">
                {when.format(new Date(p.savedAt))}
              </span>
            </button>
          </li>
        ))}
      </ul>
    </section>
  )
}
