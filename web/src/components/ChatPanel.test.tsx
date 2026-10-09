import { act, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { describe, expect, it } from 'vitest'

import type { DocumentView } from '../api/documents'
import { useChat } from '../hooks/useChat'
import { FakeSocket } from '../test/fakeSocket'
import { makeDoc, makeStep, stubApi } from '../test/fixtures'
import { ChatPanel } from './ChatPanel'

function Harness({
  initial,
  editable = false,
}: {
  initial: DocumentView
  editable?: boolean
}) {
  const [doc, setDoc] = useState(initial)
  const chat = useChat(doc.id, setDoc, FakeSocket.factory)
  return (
    <ChatPanel
      doc={doc}
      chat={chat}
      onDocument={editable ? setDoc : undefined}
    />
  )
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
      history: [makeStep('t1', { label: 'warmer please' })],
      chat: [
        { role: 'user', text: 'warmer please', step_id: null, created_at: '' },
        {
          role: 'assistant',
          text: 'Warmed it up.',
          step_id: 't1',
          created_at: '',
        },
      ],
    })
    act(() => socket.emit({ type: 'done', document: done }))
    expect(screen.queryByText('Looking at your photo…')).not.toBeInTheDocument()
    expect(screen.getByText('Warmed it up.')).toHaveClass('bubble', 'assistant')
    expect(screen.getByRole('button', { name: 'Send' })).toBeDisabled() // empty draft
  })

  it('marks replies whose edits are not in effect, and shows undo events', () => {
    const doc = makeDoc({
      history: [makeStep('t1', { label: 'warmer', active: false })],
      chat: [
        { role: 'user', text: 'warmer', step_id: null, created_at: '' },
        { role: 'assistant', text: 'Done.', step_id: 't1', created_at: '' },
        { role: 'event', text: 'Undid: warmer', step_id: null, created_at: '' },
      ],
    })
    render(<Harness initial={doc} />)
    expect(screen.getByText('not in effect')).toBeInTheDocument()
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

describe('ChatPanel plans', () => {
  it('shows a proposed plan and sends the go-ahead', async () => {
    const plan = {
      id: 'p1',
      steps: [
        { text: 'Replace the sky', kind: 'generative' as const },
        { text: 'Warm it up', kind: 'adjust' as const },
      ],
    }
    const doc = makeDoc({
      chat: [
        { role: 'user', text: 'sunset and warmer', created_at: '' },
        {
          role: 'assistant',
          text: 'Here is my plan.',
          plan,
          created_at: '',
        },
      ],
      pending_plan: plan,
    })
    render(<Harness initial={doc} />)
    expect(screen.getByText('Replace the sky')).toBeVisible()
    expect(screen.getByText('Generates new pixels')).toBeVisible()
    await userEvent.click(screen.getByRole('button', { name: 'Go ahead' }))
    act(() => FakeSocket.last!.open())
    expect(FakeSocket.last!.sent).toEqual([
      { type: 'message', text: 'Go ahead', approve_plan: true },
    ])
  })

  it('drops the button once the plan is no longer pending', () => {
    const plan = { id: 'p1', steps: [{ text: 'Sky', kind: 'ai' as const }] }
    const doc = makeDoc({
      chat: [
        { role: 'assistant', text: 'Plan.', plan, created_at: '' },
        { role: 'user', text: 'no thanks', created_at: '' },
      ],
    })
    render(<Harness initial={doc} />)
    expect(screen.getByText('Sky')).toBeVisible()
    expect(
      screen.queryByRole('button', { name: 'Go ahead' }),
    ).not.toBeInTheDocument()
  })
})

describe('ChatPanel feedback', () => {
  it('asks for a critique and sends a fix to the agent', async () => {
    const critique = {
      summary: 'A lovely moment, a little dark.',
      source: 'claude' as const,
      points: [
        {
          aspect: 'subject' as const,
          verdict: 'good' as const,
          text: 'Her smile carries it.',
          fix: null,
          fix_label: null,
        },
        {
          aspect: 'exposure' as const,
          verdict: 'improve' as const,
          text: 'The faces are in shadow.',
          fix: 'Brighten the faces a little',
          fix_label: 'Brighten faces',
        },
      ],
    }
    const after = makeDoc({
      head: 'x',
      chat: [
        {
          role: 'user',
          text: 'What do you think of this photo?',
          created_at: '',
        },
        {
          role: 'assistant',
          text: critique.summary,
          critique,
          created_at: '',
        },
      ],
    })
    stubApi({
      'POST /api/documents/abc123abc123/critique': () => Response.json(after),
    })
    render(<Harness initial={makeDoc({ head: 'x' })} editable />)
    await userEvent.click(screen.getByRole('button', { name: 'Get feedback' }))

    expect(await screen.findByText('Her smile carries it.')).toBeVisible()
    await userEvent.click(
      screen.getByRole('button', { name: 'Brighten faces' }),
    )
    act(() => FakeSocket.last!.open())
    expect(FakeSocket.last!.sent).toEqual([
      { type: 'message', text: 'Brighten the faces a little' },
    ])
  })
})
