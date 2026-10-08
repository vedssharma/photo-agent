import { useEffect, useState } from 'react'

import { api } from './api/client'
import {
  type DocumentView,
  originalUrl,
  previewUrl,
  uploadDocument,
} from './api/documents'
import { ChatPanel } from './components/ChatPanel'
import { HistoryButtons } from './components/HistoryButtons'
import { PhotoCanvas } from './components/PhotoCanvas'
import { type SocketFactory, useChat } from './hooks/useChat'
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
  return (
    <main className="workspace">
      <div className="editor">
        <div className="toolbar">
          <HistoryButtons
            doc={doc}
            onDocument={onDocument}
            disabled={chat.busy}
          />
        </div>
        <PhotoCanvas
          src={previewUrl(doc)}
          beforeSrc={originalUrl(doc)}
          alt={doc.filename}
          busy={chat.busy}
        />
      </div>
      <ChatPanel doc={doc} chat={chat} />
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
