import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import App from './App'
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
