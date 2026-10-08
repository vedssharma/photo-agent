import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import App from './App'
import {
  lastOpenProject,
  memoryProjectStore,
  rememberOpenProject,
} from './lib/projectStore'
import { FakeSocket } from './test/fakeSocket'
import { healthy, makeDoc, stubApi } from './test/fixtures'

function stubFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('App', () => {
  it('shows the backend version when the health check succeeds', async () => {
    const fetchMock = stubFetch(
      Response.json({
        status: 'ok',
        version: '1.2.3',
        anthropic_configured: true,
      }),
    )

    render(<App />)

    expect(
      await screen.findByText('Backend is up (v1.2.3).'),
    ).toBeInTheDocument()
    const request = fetchMock.mock.calls[0][0] as Request
    expect(new URL(request.url).pathname).toBe('/api/health')
    expect(screen.queryByText(/No Anthropic API key/)).not.toBeInTheDocument()
  })

  it('tells the user to set an API key when none is configured', async () => {
    stubFetch(
      Response.json({
        status: 'ok',
        version: '1.2.3',
        anthropic_configured: false,
      }),
    )

    render(<App />)

    expect(await screen.findByText(/No Anthropic API key/)).toBeInTheDocument()
  })

  it('reports when the backend is unreachable', async () => {
    stubFetch(new Response('down', { status: 502 }))

    render(<App />)

    expect(
      await screen.findByText(/Backend unreachable: Error: HTTP 502/),
    ).toBeInTheDocument()
  })
})

describe('App with a photo', () => {
  it('uploads a picked photo and shows its preview', async () => {
    const doc = makeDoc()
    const fetchMock = stubApi({
      'GET /api/health': healthy,
      'POST /api/documents': () => Response.json(doc, { status: 201 }),
    })

    render(<App />)
    await userEvent.upload(
      screen.getByLabelText('Photo file'),
      new File(['x'], 'cat.heic', { type: '' }),
    )

    const img = await screen.findByRole('img', { name: 'cat.heic' })
    expect(img).toHaveAttribute(
      'src',
      '/api/documents/abc123abc123/preview?revision=original',
    )
    const upload = fetchMock.mock.calls.find(([r]) => r.method === 'POST')![0]
    expect((await upload.formData()).get('file')).toBeTruthy()
  })

  it('explains why a photo could not be opened', async () => {
    stubApi({
      'GET /api/health': healthy,
      'POST /api/documents': () =>
        Response.json(
          { detail: 'Could not read this file as an image.' },
          { status: 415 },
        ),
    })

    render(<App />)
    await userEvent.upload(
      screen.getByLabelText('Photo file'),
      new File(['x'], 'bad.jpg'),
    )

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Could not open bad.jpg: Could not read this file as an image.',
    )
  })
})

describe('Editor', () => {
  async function openPhoto() {
    stubApi({
      'GET /api/health': healthy,
      'POST /api/documents': () => Response.json(makeDoc(), { status: 201 }),
    })
    render(<App createSocket={FakeSocket.factory} />)
    await userEvent.upload(
      screen.getByLabelText('Photo file'),
      new File(['x'], 'cat.heic'),
    )
    await screen.findByRole('img', { name: 'cat.heic' })
  }

  it('opens the download dialog from the toolbar', async () => {
    await openPhoto()
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))
    expect(
      screen.getByRole('dialog', { name: 'Download photo' }),
    ).toBeInTheDocument()
  })

  it('refreshes the photo when the agent finishes a turn', async () => {
    await openPhoto()
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Message' }),
      'warmer{Enter}',
    )
    const socket = FakeSocket.last!
    act(() => socket.open())
    expect(screen.getByRole('button', { name: 'Download' })).toBeDisabled()

    act(() =>
      socket.emit({
        type: 'done',
        document: makeDoc({ revision: 'turn1', can_undo: true }),
      }),
    )

    expect(screen.getByRole('img', { name: 'cat.heic' })).toHaveAttribute(
      'src',
      '/api/documents/abc123abc123/preview?revision=turn1',
    )
    expect(screen.getByRole('button', { name: 'Undo' })).toBeEnabled()
  })
})

describe('Projects', () => {
  it('reopens the project that was open when the page closed', async () => {
    rememberOpenProject('abc123abc123')
    stubApi({
      'GET /api/health': healthy,
      'GET /api/documents/abc123abc123': () =>
        Response.json(makeDoc({ filename: 'trip.jpg' })),
    })
    render(<App projectStore={memoryProjectStore()} />)
    expect(
      await screen.findByRole('img', { name: 'trip.jpg' }),
    ).toBeInTheDocument()
  })

  it('restores from the browser copy when the server no longer has it', async () => {
    const store = memoryProjectStore()
    await store.put({
      id: 'abc123abc123',
      filename: 'trip.jpg',
      savedAt: 1,
      graph: '{"id":"abc123abc123"}',
      original: new Blob(['jpeg bytes']),
    })
    const fetchMock = stubApi({
      'GET /api/health': healthy,
      'GET /api/documents/abc123abc123': () =>
        Response.json({ detail: 'No such document.' }, { status: 404 }),
      'POST /api/projects': () =>
        Response.json(makeDoc({ filename: 'trip.jpg' }), { status: 201 }),
    })
    render(<App projectStore={store} />)

    await userEvent.click(
      await screen.findByRole('button', { name: /trip\.jpg/ }),
    )

    expect(
      await screen.findByRole('img', { name: 'trip.jpg' }),
    ).toBeInTheDocument()
    const post = fetchMock.mock.calls.find(([r]) => r.method === 'POST')![0]
    const form = await post.formData()
    expect(form.get('graph')).toBe('{"id":"abc123abc123"}')
    expect(form.get('original')).toBeTruthy()
  })

  it('opens a saved project file', async () => {
    const fetchMock = stubApi({
      'GET /api/health': healthy,
      'POST /api/projects': () =>
        Response.json(makeDoc({ filename: 'trip.jpg' }), { status: 201 }),
    })
    render(<App projectStore={memoryProjectStore()} />)
    await userEvent.upload(
      screen.getByLabelText('Project file'),
      new File(['zip'], 'trip.photoagent'),
    )
    expect(
      await screen.findByRole('img', { name: 'trip.jpg' }),
    ).toBeInTheDocument()
    const post = fetchMock.mock.calls.find(([r]) => r.method === 'POST')![0]
    expect((await post.formData()).get('file')).toBeTruthy()
  })

  it('keeps a copy of the open project in the browser', async () => {
    const store = memoryProjectStore()
    stubApi({
      'GET /api/health': healthy,
      'POST /api/documents': () => Response.json(makeDoc(), { status: 201 }),
      'GET /api/documents/abc123abc123/graph': () =>
        new Response('{"saved":true}'),
      'GET /api/documents/abc123abc123/source': () => new Response('bytes'),
    })
    render(<App projectStore={store} />)
    await userEvent.upload(
      screen.getByLabelText('Photo file'),
      new File(['x'], 'cat.heic'),
    )
    await screen.findByRole('img', { name: 'cat.heic' })
    await waitFor(
      async () =>
        expect((await store.get('abc123abc123'))?.graph).toBe('{"saved":true}'),
      { timeout: 3000 },
    )
    expect(lastOpenProject()).toBe('abc123abc123')

    await userEvent.click(screen.getByRole('button', { name: 'Open another' }))
    expect(lastOpenProject()).toBeNull()
    expect(
      await screen.findByRole('button', { name: /cat\.heic/ }),
    ).toBeInTheDocument()
  })
})
