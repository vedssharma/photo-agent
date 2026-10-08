import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'

import type { DocumentView } from '../api/documents'
import type { Chat } from '../hooks/useChat'

const SUGGESTIONS = [
  'Make it pop a little more',
  'Fix the lighting',
  'Make this look warmer and less washed out',
  'Crop it for Instagram',
]

interface Props {
  doc: DocumentView
  chat: Chat
}

export function ChatPanel({ doc, chat }: Props) {
  const [draft, setDraft] = useState('')
  const log = useRef<HTMLOListElement>(null)
  const applied = new Set(doc.turns.filter((t) => t.applied).map((t) => t.id))

  useEffect(() => {
    log.current?.lastElementChild?.scrollIntoView?.({ block: 'end' })
  }, [doc.chat.length, chat.pending])

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (chat.busy || !draft.trim()) return
    chat.send(draft)
    setDraft('')
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing)
      submit(e)
  }

  const empty = doc.chat.length === 0 && !chat.pending
  return (
    <aside className="chat" aria-label="Chat with the editor">
      <ol className="chat-log" ref={log} aria-live="polite">
        {empty && (
          <li className="chat-hint">
            <p>Tell me how you would like this photo to look.</p>
            <p className="fine-print">
              A downsized copy of the photo is sent to Claude so it can see what
              to change. The original stays on this computer.
            </p>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => chat.send(s)}>
                  {s}
                </button>
              ))}
            </div>
          </li>
        )}
        {doc.chat.map((entry, i) =>
          entry.role === 'event' ? (
            <li key={i} className="chat-event">
              {entry.text}
            </li>
          ) : (
            <li key={i} className={`bubble ${entry.role}`}>
              {entry.text}
              {entry.turn_id && !applied.has(entry.turn_id) && (
                <span className="tag">undone</span>
              )}
            </li>
          ),
        )}
        {chat.pending && (
          <>
            <li className="bubble user">{chat.pending.request}</li>
            <li className="bubble assistant working">
              {chat.pending.operations.length > 0 && (
                <ul className="ops">
                  {chat.pending.operations.map((op, i) => (
                    <li key={i}>{op}</li>
                  ))}
                </ul>
              )}
              {chat.pending.reply || (
                <span className="thinking">Looking at your photo…</span>
              )}
            </li>
          </>
        )}
      </ol>
      {chat.error && (
        <p className="error chat-error" role="alert">
          {chat.error}
        </p>
      )}
      <form className="chat-input" onSubmit={submit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Describe a change, e.g. “brighter, and warmer skin tones”"
          aria-label="Message"
          rows={3}
        />
        <button
          type="submit"
          className="primary"
          disabled={chat.busy || !draft.trim()}
        >
          {chat.busy ? 'Editing…' : 'Send'}
        </button>
      </form>
    </aside>
  )
}
