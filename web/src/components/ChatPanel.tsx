import { useEffect, useRef, useState } from 'react'
import type { FormEvent, KeyboardEvent } from 'react'

import type { DocumentView } from '../api/documents'
import type { Chat } from '../hooks/useChat'
import { critiquePhoto } from '../api/advice'
import { PHOTO_TYPES } from '../api/documents'
import {
  type Reference,
  referenceUrl,
  uploadReference,
} from '../api/references'
import { CritiqueCard } from './CritiqueCard'
import { PlanCard } from './PlanCard'
import { SuggestionPicker } from './SuggestionPicker'

const SUGGESTIONS = [
  'Make it pop a little more',
  'Fix the lighting',
  'Make this look warmer and less washed out',
  'Crop it for Instagram',
]

interface Props {
  doc: DocumentView
  chat: Chat
  /** Takes an updated document, after a suggestion is picked. */
  onDocument?: (doc: DocumentView) => void
}

export function ChatPanel({ doc, chat, onDocument }: Props) {
  const [draft, setDraft] = useState('')
  const [critiquing, setCritiquing] = useState(false)
  const [critiqueError, setCritiqueError] = useState<string | null>(null)
  const [attached, setAttached] = useState<Reference[]>([])
  const [attaching, setAttaching] = useState(false)
  const picker = useRef<HTMLInputElement>(null)
  const log = useRef<HTMLOListElement>(null)
  const active = new Set(doc.history.filter((s) => s.active).map((s) => s.id))

  // Manual edits are listed in the history panel; the agent still hears about them.
  const shown = doc.chat
    .map((entry, i) => ({ entry, i }))
    .filter(({ entry }) => !(entry.role === 'event' && entry.step_id))

  useEffect(() => {
    log.current?.lastElementChild?.scrollIntoView?.({ block: 'end' })
  }, [shown.length, chat.pending])

  function submit(e?: FormEvent) {
    e?.preventDefault()
    if (chat.busy || !draft.trim()) return
    chat.send(draft, { references: attached.map((r) => r.id) })
    setDraft('')
    setAttached([])
  }

  async function attach(file: File) {
    setAttaching(true)
    setCritiqueError(null)
    try {
      const ref = await uploadReference(doc.id, file)
      setAttached((all) => [...all, ref].slice(-4))
    } catch (err) {
      setCritiqueError(`Could not add ${file.name}: ${(err as Error).message}`)
    } finally {
      setAttaching(false)
    }
  }

  async function critique() {
    if (!onDocument) return
    setCritiquing(true)
    setCritiqueError(null)
    try {
      onDocument(await critiquePhoto(doc.id))
    } catch (err) {
      setCritiqueError(`Could not get feedback: ${(err as Error).message}`)
    } finally {
      setCritiquing(false)
    }
  }

  function onKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing)
      submit(e)
  }

  const empty = shown.length === 0 && !chat.pending && !critiquing
  return (
    <aside className="chat" aria-label="Chat with the editor">
      <ol className="chat-log" ref={log} aria-live="polite">
        {empty && (
          <li className="chat-hint">
            {onDocument && doc.head === null && (
              <SuggestionPicker
                doc={doc}
                onDocument={onDocument}
                disabled={chat.busy}
              />
            )}
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
        {shown.map(({ entry, i }) =>
          entry.role === 'event' ? (
            <li key={i} className="chat-event">
              {entry.text}
            </li>
          ) : (
            <li key={i} className={`bubble ${entry.role}`}>
              {entry.references && entry.references.length > 0 && (
                <span className="bubble-refs">
                  {entry.references.map((id) => (
                    <img
                      key={id}
                      src={referenceUrl(doc.id, id)}
                      alt="Reference photo"
                    />
                  ))}
                </span>
              )}
              {entry.text}
              {entry.step_id && !active.has(entry.step_id) && (
                <span className="tag">not in effect</span>
              )}
              {entry.critique && (
                <CritiqueCard
                  critique={entry.critique}
                  disabled={chat.busy}
                  onFix={(request) => chat.send(request)}
                />
              )}
              {entry.plan && (
                <PlanCard
                  plan={entry.plan}
                  pending={doc.pending_plan?.id === entry.plan.id}
                  disabled={chat.busy}
                  onApprove={() => chat.send('Go ahead', { approvePlan: true })}
                />
              )}
            </li>
          ),
        )}
        {critiquing && (
          <li className="bubble assistant working">
            <span className="thinking">Taking a good look at your photo…</span>
          </li>
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
      {(chat.error ?? critiqueError) && (
        <p className="error chat-error" role="alert">
          {chat.error ?? critiqueError}
        </p>
      )}
      <div className="chat-tools">
        {onDocument && (
          <button
            type="button"
            disabled={chat.busy || critiquing}
            onClick={() => void critique()}
            title="What works in this photo, what does not, and a fix for each"
          >
            Get feedback
          </button>
        )}
        <button
          type="button"
          disabled={chat.busy || attaching}
          onClick={() => picker.current?.click()}
          title="Share another photo, then ask to make this one look like it"
        >
          {attaching ? 'Adding…' : 'Add a reference photo'}
        </button>
        <input
          ref={picker}
          type="file"
          accept={PHOTO_TYPES}
          hidden
          aria-label="Reference photo file"
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            if (file) void attach(file)
          }}
        />
      </div>
      {attached.length > 0 && (
        <ul className="attached" aria-label="Reference photos to send">
          {attached.map((ref) => (
            <li key={ref.id}>
              <img src={referenceUrl(doc.id, ref.id)} alt="" />
              {ref.filename}
              <button
                type="button"
                className="icon"
                aria-label={`Remove ${ref.filename}`}
                onClick={() =>
                  setAttached((all) => all.filter((r) => r.id !== ref.id))
                }
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      <form className="chat-input" onSubmit={submit}>
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={
            attached.length
              ? 'Say what to take from it, e.g. “make mine look like this”'
              : 'Describe a change, e.g. “brighter, and warmer skin tones”'
          }
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
