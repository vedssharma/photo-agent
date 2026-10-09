import { useCallback, useEffect, useState } from 'react'

import { api } from './api/client'
import {
  type DocumentView,
  beforeUrl,
  downloadProject,
  fetchDocument,
  isProjectFile,
  layerMaskUrl,
  openProjectFile,
  previewUrl,
  restoreProject,
  saveFile,
  uploadDocument,
} from './api/documents'
import { ChatPanel } from './components/ChatPanel'
import { ExportDialog } from './components/ExportDialog'
import { HistoryButtons } from './components/HistoryButtons'
import { HistoryPanel } from './components/HistoryPanel'
import { LayersPanel } from './components/LayersPanel'
import { LivePreview } from './components/LivePreview'
import { MaskOverlay } from './components/MaskOverlay'
import { PhotoCanvas } from './components/PhotoCanvas'
import { type SocketFactory, useChat } from './hooks/useChat'
import { jobText, useJobs } from './hooks/useJobs'
import { useManualEdit } from './hooks/useManualEdit'
import { useOperationSpecs } from './hooks/useOperationSpecs'
import { DEFAULT_MASK_TOOL, type MaskTool, isDrawn } from './lib/masks'
import { liveTarget } from './lib/livePreview'
import { type Preview, updateLayer } from './lib/state'
import { PhotoPicker } from './components/PhotoPicker'
import { RecentProjects } from './components/RecentProjects'
import { RecipesPanel } from './components/RecipesPanel'
import { useAutosave } from './hooks/useAutosave'
import {
  type ProjectStore,
  type ProjectSummary,
  indexedDbProjectStore,
  lastOpenProject,
  rememberOpenProject,
} from './lib/projectStore'

type Health =
  | { state: 'loading' }
  | { state: 'ok'; version: string; anthropicConfigured: boolean }
  | { state: 'error'; message: string }

function useHealth(): Health {
  const [health, setHealth] = useState<Health>({ state: 'loading' })

  useEffect(() => {
    api
      .GET('/api/health')
      .then(({ data, error, response }) => {
        if (error || !data) throw new Error(`HTTP ${response.status}`)
        setHealth({
          state: 'ok',
          version: data.version,
          anthropicConfigured: data.anthropic_configured,
        })
      })
      .catch((err: unknown) => {
        setHealth({ state: 'error', message: String(err) })
      })
  }, [])

  return health
}

function Editor({
  doc,
  onDocument,
  createSocket,
  projectStore,
}: {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  createSocket?: SocketFactory
  projectStore: ProjectStore
}) {
  const chat = useChat(doc.id, onDocument, createSocket)
  const [exporting, setExporting] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  useAutosave(doc, projectStore, setSaveError)

  async function saveProject() {
    setSaveError(null)
    try {
      saveFile(await downloadProject(doc))
    } catch (err) {
      setSaveError(`Could not save the project: ${(err as Error).message}`)
    }
  }

  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const [maskTool, setMaskTool] = useState<MaskTool>(DEFAULT_MASK_TOOL)
  const manual = useManualEdit(doc, onDocument)
  const specs = useOperationSpecs()
  const locked = chat.busy
  const job = useJobs(doc.id, chat.busy || manual.working)
  const layer = doc.state.layers.find((l) => l.id === selectedLayer) ?? null
  const mask = layer?.mask ?? null
  const showOverlay = layer && mask && (isDrawn(mask.kind) || maskTool.show)
  // A slider being dragged, shown instantly until the server's render of it arrives.
  const [preview, setPreview] = useState<
    (Preview & { revision: string }) | null
  >(null)
  const live =
    preview && preview.revision === doc.revision
      ? liveTarget(doc.state, preview)
      : null
  return (
    <main className="workspace">
      <aside className="side" aria-label="Edits">
        <LayersPanel
          doc={doc}
          onEdit={manual.edit}
          disabled={locked}
          selected={selectedLayer}
          onSelect={setSelectedLayer}
          maskTool={maskTool}
          onMaskTool={setMaskTool}
          specs={specs}
          onPreview={(p) => setPreview(p && { ...p, revision: doc.revision })}
        />
        <RecipesPanel
          doc={doc}
          onDocument={onDocument}
          disabled={chat.busy || manual.working}
        />
        <HistoryPanel
          doc={doc}
          onDocument={onDocument}
          disabled={chat.busy || manual.working}
        />
      </aside>
      <div className="editor">
        <div className="toolbar">
          <HistoryButtons
            doc={doc}
            onDocument={onDocument}
            disabled={chat.busy || manual.working}
          />
          {(manual.error ?? saveError) && (
            <span className="error" role="alert">
              {manual.error ?? saveError}
            </span>
          )}
          <span className="spacer" />
          <button
            type="button"
            onClick={saveProject}
            disabled={chat.busy}
            title="Download the original and all edits as one file you can reopen later"
          >
            Save project
          </button>
          <button
            type="button"
            className="primary"
            onClick={() => setExporting(true)}
            disabled={chat.busy}
          >
            Download
          </button>
        </div>
        <PhotoCanvas
          src={previewUrl(doc)}
          beforeSrc={beforeUrl(doc)}
          alt={doc.filename}
          busy={chat.busy || job !== null}
          status={job && jobText(job)}
          overlay={
            showOverlay || live
              ? (frame) => (
                  <>
                    {live && (
                      <LivePreview
                        src={previewUrl(doc)}
                        maskSrc={
                          live.maskLayerId
                            ? layerMaskUrl(doc, live.maskLayerId)
                            : undefined
                        }
                        adjustment={live.adjustment}
                        opacity={live.opacity}
                      />
                    )}
                    {showOverlay && (
                      <MaskOverlay
                        key={`${layer.id}:${doc.revision}`}
                        mask={mask}
                        width={frame.width}
                        height={frame.height}
                        tool={maskTool}
                        maskSrc={
                          maskTool.show
                            ? layerMaskUrl(doc, layer.id)
                            : undefined
                        }
                        onChange={(next, label) =>
                          manual.edit(
                            `${label} on “${layer.name}”`,
                            (s) => updateLayer(s, layer.id, { mask: next }),
                            `layer:${layer.id}:mask:draw`,
                          )
                        }
                      />
                    )}
                  </>
                )
              : undefined
          }
        />
      </div>
      <ChatPanel doc={doc} chat={chat} />
      {exporting && (
        <ExportDialog doc={doc} onClose={() => setExporting(false)} />
      )}
    </main>
  )
}

/** A project from the server if it still has it, else from the browser's copy. */
async function findProject(
  id: string,
  store: ProjectStore,
): Promise<DocumentView> {
  const found = await fetchDocument(id)
  if (found) return found
  const saved = await store.get(id)
  if (!saved) throw new Error('it is no longer saved anywhere')
  return restoreProject(saved.original, saved.filename, saved.graph)
}

interface AppProps {
  createSocket?: SocketFactory
  /** Where projects are kept in the browser; IndexedDB by default. */
  projectStore?: ProjectStore
}

function App({ createSocket, projectStore }: AppProps = {}) {
  const health = useHealth()
  const [store] = useState(() => projectStore ?? indexedDbProjectStore())
  const [doc, setDoc] = useState<DocumentView | null>(null)
  const [opening, setOpening] = useState(() => lastOpenProject() !== null)
  const [openError, setOpenError] = useState<string | null>(null)
  const [recent, setRecent] = useState<ProjectSummary[]>([])

  const show = useCallback((next: DocumentView | null) => {
    setDoc(next)
    rememberOpenProject(next?.id ?? null)
  }, [])

  async function open(file: File) {
    setOpening(true)
    setOpenError(null)
    try {
      show(
        await (isProjectFile(file)
          ? openProjectFile(file)
          : uploadDocument(file)),
      )
    } catch (err) {
      setOpenError(`Could not open ${file.name}: ${(err as Error).message}`)
    } finally {
      setOpening(false)
    }
  }

  const reopen = useCallback(
    (id: string) =>
      findProject(id, store)
        .then(show, (err: Error) => {
          rememberOpenProject(null)
          setOpenError(`Could not reopen the project: ${err.message}`)
        })
        .finally(() => setOpening(false)),
    [store, show],
  )

  function resume(id: string) {
    setOpening(true)
    setOpenError(null)
    void reopen(id)
  }

  // Pick up where the last visit left off.
  useEffect(() => {
    const last = lastOpenProject()
    if (last) void reopen(last)
  }, [reopen])

  useEffect(() => {
    if (doc) return
    let live = true
    store
      .list()
      .then((projects) => {
        if (live) setRecent(projects)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [doc, store])

  return (
    <div className="app">
      <header className="topbar">
        <h1>photo-agent</h1>
        {doc && (
          <>
            <span className="filename">{doc.filename}</span>
            <div className="spacer" />
            <button type="button" onClick={() => show(null)}>
              Open another
            </button>
          </>
        )}
      </header>

      {doc ? (
        <Editor
          key={doc.id}
          doc={doc}
          onDocument={setDoc}
          createSocket={createSocket}
          projectStore={store}
        />
      ) : (
        <main className="landing">
          <p>Describe the edit you want, and the agent does the rest.</p>
          <PhotoPicker onPick={open} busy={opening} error={openError} />
          <RecentProjects projects={recent} busy={opening} onOpen={resume} />
          <p className="status">
            {health.state === 'loading' && 'Connecting to the backend…'}
            {health.state === 'ok' && `Backend is up (v${health.version}).`}
            {health.state === 'error' &&
              `Backend unreachable: ${health.message}`}
          </p>
          {health.state === 'ok' && !health.anthropicConfigured && (
            <p className="status">
              No Anthropic API key found. Set <code>ANTHROPIC_API_KEY</code> in{' '}
              <code>.env</code> at the repo root.
            </p>
          )}
        </main>
      )}
    </div>
  )
}

export default App
