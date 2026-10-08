import { useEffect, useState } from 'react'

import { api } from './api/client'
import {
  type DocumentView,
  beforeUrl,
  layerMaskUrl,
  previewUrl,
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
import { useManualEdit } from './hooks/useManualEdit'
import { useOperationSpecs } from './hooks/useOperationSpecs'
import { DEFAULT_MASK_TOOL, type MaskTool, isDrawn } from './lib/masks'
import { liveTarget } from './lib/livePreview'
import { type Preview, updateLayer } from './lib/state'
import { PhotoPicker } from './components/PhotoPicker'

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
}: {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  createSocket?: SocketFactory
}) {
  const chat = useChat(doc.id, onDocument, createSocket)
  const [exporting, setExporting] = useState(false)
  const [selectedLayer, setSelectedLayer] = useState<string | null>(null)
  const [maskTool, setMaskTool] = useState<MaskTool>(DEFAULT_MASK_TOOL)
  const manual = useManualEdit(doc, onDocument)
  const specs = useOperationSpecs()
  const locked = chat.busy
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
          {manual.error && (
            <span className="error" role="alert">
              {manual.error}
            </span>
          )}
          <span className="spacer" />
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
          busy={chat.busy}
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

function App({ createSocket }: { createSocket?: SocketFactory } = {}) {
  const health = useHealth()
  const [doc, setDoc] = useState<DocumentView | null>(null)
  const [opening, setOpening] = useState(false)
  const [openError, setOpenError] = useState<string | null>(null)

  async function open(file: File) {
    setOpening(true)
    setOpenError(null)
    try {
      setDoc(await uploadDocument(file))
    } catch (err) {
      setOpenError(`Could not open ${file.name}: ${(err as Error).message}`)
    } finally {
      setOpening(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>photo-agent</h1>
        {doc && (
          <>
            <span className="filename">{doc.filename}</span>
            <div className="spacer" />
            <button type="button" onClick={() => setDoc(null)}>
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
        />
      ) : (
        <main className="landing">
          <p>Describe the edit you want, and the agent does the rest.</p>
          <PhotoPicker onPick={open} busy={opening} error={openError} />
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
