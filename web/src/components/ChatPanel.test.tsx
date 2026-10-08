import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'

import type { DocumentView } from '../api/documents'
import { useChat } from '../hooks/useChat'
import { FakeSocket } from '../test/fakeSocket'
import { makeDoc } from '../test/fixtures'
import { ChatPanel } from './ChatPanel'

function Harness({ initial }: { initial: DocumentView }) {
  const [doc, setDoc] = useState(initial)
  const chat = useChat(doc.id, setDoc, FakeSocket.factory)
  return <ChatPanel doc={doc} chat={chat} />
}

async function sendMessage(text: string) {
  await userEvent.type(
    screen.getByRole('textbox', { name: 'Message' }),
    `${text}{Enter}`,
  )
  const socket = FakeSocket.last!
  act(() => socket.open())
  return socket
}

describe('ChatPanel', () => {
  it('offers suggestions on an empty conversation', async () => {
    render(<Harness initial={makeDoc()} />)
    await userEvent.click(
      screen.getByRole('button', { name: 'Fix the lighting' }),
    )
    act(() => FakeSocket.last!.open())
    expect(FakeSocket.last!.sent).toEqual([
      { type: 'message', text: 'Fix the lighting' },
    ])
  })

  it('streams the reply and the edits, then shows the saved conversation', async () => {
    render(<Harness initial={makeDoc()} />)
    const socket = await sendMessage('warmer please')

    expect(socket.url).toBe(
      'ws://localhost:3000/api/documents/abc123abc123/chat',
    )
    expect(socket.sent).toEqual([{ type: 'message', text: 'warmer please' }])
    expect(screen.getByText('Looking at your photo…')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Editing…' })).toBeDisabled()

    act(() => {
      socket.emit({ type: 'turn_started' })
      socket.emit({
        type: 'operation',
        action: 'added',
        summary: 'White balance (temperature +20)',
      })
      socket.emit({ type: 'text', text: 'Warmed ' })
      socket.emit({ type: 'text', text: 'it up.' })
    })
    expect(
      screen.getByText('White balance (temperature +20)'),
    ).toBeInTheDocument()
    expect(screen.getByText('Warmed it up.')).toBeInTheDocument()

    const done = makeDoc({
      revision: 't1',
      turns: [
        {
          id: 't1',
          request: 'warmer please',
          reply: 'Warmed it up.',
          operations: [],
          applied: true,
        },
      ],
      chat: [
        { role: 'user', text: 'warmer please', turn_id: null, created_at: '' },
        {
          role: 'assistant',
          text: 'Warmed it up.',
          turn_id: 't1',
          created_at: '',
        },
      ],
    })
    act(() => socket.emit({ type: 'done', document: done }))
    expect(screen.queryByText('Looking at your photo…')).not.toBeInTheDocument()
    expect(screen.getByText('Warmed it up.')).toHaveClass('bubble', 'assistant')
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled() // empty draft
  })

  it('marks replies whose edits were undone, and shows undo events', () => {
    const doc = makeDoc({
      turns: [
        {
          id: 't1',
          request: 'warmer',
          reply: 'Done.',
          operations: [],
          applied: false,
        },
      ],
      chat: [
        { role: 'user', text: 'warmer', turn_id: null, created_at: '' },
        { role: 'assistant', text: 'Done.', turn_id: 't1', created_at: '' },
        { role: 'event', text: 'Undid: warmer', turn_id: null, created_at: '' },
      ],
    })
    render(<Harness initial={doc} />)
    expect(screen.getByText('undone')).toBeInTheDocument()
    expect(screen.getByText('Undid: warmer')).toHaveClass('chat-event')
  })

  it('shows agent errors and lets the user try again', async () => {
    render(<Harness initial={makeDoc()} />)
    const socket = await sendMessage('brighter')
    act(() =>
      socket.emit({
        type: 'error',
        message: 'The agent needs a Claude API key.',
      }),
    )
    expect(screen.getByRole('alert')).toHaveTextContent(
      'needs a Claude API key',
    )
    expect(screen.getByRole('textbox', { name: 'Message' })).toBeEnabled()
  })

  it('reports a dropped connection mid-turn and reconnects on the next message', async () => {
    render(<Harness initial={makeDoc()} />)
    const first = await sendMessage('brighter')
    act(() => first.drop())
    expect(screen.getByRole('alert')).toHaveTextContent('Lost the connection')

    const second = await sendMessage('brighter')
    expect(second).not.toBe(first)
    expect(second.sent).toHaveLength(1)
  })

  it('sends with Enter and keeps Shift+Enter for new lines', async () => {
    render(<Harness initial={makeDoc()} />)
    const box = screen.getByRole('textbox', { name: 'Message' })
    await userEvent.type(box, 'line one{Shift>}{Enter}{/Shift}line two')
    expect(box).toHaveValue('line one\nline two')
  })
})
