import { useCallback, useEffect, useRef, useState } from 'react'

import type { DocumentView } from '../api/documents'
import type { components } from '../api/schema'

export type ChatEvent = components['schemas']['ChatEvent']
export type ChatMessage = components['schemas']['ChatMessage']

/** The minimal WebSocket surface the hook needs, so tests can supply a fake. */
export interface ChatSocket {
  readonly readyState: number
  send(data: string): void
  close(): void
  onopen: ((ev: Event) => void) | null
  onmessage: ((ev: MessageEvent) => void) | null
  onclose: ((ev: CloseEvent) => void) | null
  onerror: ((ev: Event) => void) | null
}

export type SocketFactory = (url: string) => ChatSocket

const OPEN = 1

export function chatUrl(
  docId: string,
  location: Location = window.location,
): string {
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${location.host}/api/documents/${docId}/chat`
}

/** What the agent is doing in the turn that is currently running. */
export interface PendingTurn {
  request: string
  reply: string
  operations: string[]
}

export interface Chat {
  pending: PendingTurn | null
  error: string | null
  busy: boolean
  /** Send a message; with `approvePlan`, it goes ahead with the plan just proposed. */
  send: (text: string, options?: SendOptions) => void
}

export interface SendOptions {
  approvePlan?: boolean
  /** Ids of reference photos shared with the message. */
  references?: string[]
}

/**
 * Talks to the agent over the chat WebSocket for one document. Streams the reply into
 * `pending` and hands the updated document to `onDocument` when the turn finishes.
 */
export function useChat(
  docId: string,
  onDocument: (doc: DocumentView) => void,
  createSocket: SocketFactory = (url) => new WebSocket(url),
): Chat {
  const [pending, setPending] = useState<PendingTurn | null>(null)
  const [error, setError] = useState<string | null>(null)
  const socket = useRef<ChatSocket | null>(null)
  const outbox = useRef<string[]>([])
  const waiting = useRef(false)
  const onDocumentRef = useRef(onDocument)
  const factory = useRef(createSocket)
  useEffect(() => {
    onDocumentRef.current = onDocument
  })

  useEffect(
    () => () => {
      socket.current?.close()
      socket.current = null
    },
    [docId],
  )

  const handle = useCallback((event: ChatEvent) => {
    switch (event.type) {
      case 'turn_started':
        break
      case 'text':
        setPending((p) => p && { ...p, reply: p.reply + event.text })
        break
      case 'operation':
        setPending(
          (p) => p && { ...p, operations: [...p.operations, event.summary] },
        )
        break
      case 'done':
        waiting.current = false
        onDocumentRef.current(event.document)
        setPending(null)
        break
      case 'error':
        waiting.current = false
        setError(event.message)
        setPending(null)
        break
    }
  }, [])

  const connect = useCallback((): ChatSocket => {
    const ws = factory.current(chatUrl(docId))
    ws.onopen = () => {
      for (const message of outbox.current.splice(0)) ws.send(message)
    }
    ws.onmessage = (e) => handle(JSON.parse(String(e.data)) as ChatEvent)
    ws.onclose = () => {
      if (socket.current !== ws) return
      socket.current = null
      if (waiting.current) {
        waiting.current = false
        setError('Lost the connection to the agent. Please try again.')
        setPending(null)
      }
    }
    ws.onerror = () => {}
    socket.current = ws
    return ws
  }, [docId, handle])

  const send = useCallback(
    (text: string, options: SendOptions = {}) => {
      const request = text.trim()
      if (!request) return
      setError(null)
      waiting.current = true
      setPending({ request, reply: '', operations: [] })
      // approve_plan defaults to false on the server, so leave it out unless set.
      const message: Partial<ChatMessage> = { type: 'message', text: request }
      if (options.approvePlan) message.approve_plan = true
      if (options.references?.length) message.references = options.references
      const payload = JSON.stringify(message)
      const ws = socket.current ?? connect()
      if (ws.readyState === OPEN) ws.send(payload)
      else outbox.current.push(payload)
    },
    [connect],
  )

  return { pending, error, busy: pending !== null, send }
}
