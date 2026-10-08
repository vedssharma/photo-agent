/**
 * Browser storage for projects, so closing the tab (or losing the server's data folder)
 * never loses work. Each project keeps the original photo and the latest edit graph.
 */

export interface SavedProject {
  id: string
  filename: string
  /** When the edits were last saved, in ms since the epoch. */
  savedAt: number
  /** The stored document JSON, exactly as the server keeps it. */
  graph: string
  original: Blob
}

export type ProjectSummary = Pick<SavedProject, 'id' | 'filename' | 'savedAt'>

export interface ProjectStore {
  /** Newest first. */
  list(): Promise<ProjectSummary[]>
  get(id: string): Promise<SavedProject | undefined>
  put(project: SavedProject): Promise<void>
  remove(id: string): Promise<void>
}

/** How many projects to keep; the least recently saved are dropped first. */
export const MAX_PROJECTS = 8

const LAST_OPEN_KEY = 'photo-agent:open-project'

/** Remember which project is open, to reopen it when the page loads again. */
export function rememberOpenProject(id: string | null) {
  try {
    if (id) localStorage.setItem(LAST_OPEN_KEY, id)
    else localStorage.removeItem(LAST_OPEN_KEY)
  } catch {
    // Storage can be unavailable (private windows); reopening is a convenience.
  }
}

export function lastOpenProject(): string | null {
  try {
    return localStorage.getItem(LAST_OPEN_KEY)
  } catch {
    return null
  }
}

function newestFirst(projects: ProjectSummary[]): ProjectSummary[] {
  return projects
    .map(({ id, filename, savedAt }) => ({ id, filename, savedAt }))
    .sort((a, b) => b.savedAt - a.savedAt)
}

/** An in-memory store, for tests and for browsers without IndexedDB. */
export function memoryProjectStore(): ProjectStore {
  const items = new Map<string, SavedProject>()
  return {
    list: async () => newestFirst([...items.values()]),
    get: async (id) => items.get(id),
    put: async (project) => {
      items.set(project.id, project)
      for (const old of newestFirst([...items.values()]).slice(MAX_PROJECTS))
        items.delete(old.id)
    },
    remove: async (id) => {
      items.delete(id)
    },
  }
}

const DB_NAME = 'photo-agent'
const STORE = 'projects'

function request<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

/** Projects kept in IndexedDB, which can hold large photos. */
export function indexedDbProjectStore(): ProjectStore {
  if (typeof indexedDB === 'undefined') return memoryProjectStore()
  let db: Promise<IDBDatabase> | null = null
  const open = () => {
    db ??= new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1)
      req.onupgradeneeded = () =>
        req.result.createObjectStore(STORE, { keyPath: 'id' })
      req.onsuccess = () => resolve(req.result)
      req.onerror = () => reject(req.error)
    })
    return db
  }
  const objects = async (mode: IDBTransactionMode) =>
    (await open()).transaction(STORE, mode).objectStore(STORE)

  const list = async () =>
    newestFirst(
      await request<SavedProject[]>((await objects('readonly')).getAll()),
    )

  return {
    list,
    get: async (id) =>
      request<SavedProject | undefined>((await objects('readonly')).get(id)),
    put: async (project) => {
      await request((await objects('readwrite')).put(project))
      const stale = (await list()).slice(MAX_PROJECTS)
      for (const old of stale)
        await request((await objects('readwrite')).delete(old.id))
    },
    remove: async (id) => {
      await request((await objects('readwrite')).delete(id))
    },
  }
}
