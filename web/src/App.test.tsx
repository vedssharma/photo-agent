import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import App from './App'

function stubFetch(response: Response) {
  const fetchMock = vi.fn().mockResolvedValue(response)
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

describe('App', () => {
  it('shows the backend version when the health check succeeds', async () => {
    const fetchMock = stubFetch(
      Response.json({ status: 'ok', version: '1.2.3' }),
    )

    render(<App />)

    expect(
      await screen.findByText('Backend is up (v1.2.3).'),
    ).toBeInTheDocument()
    const request = fetchMock.mock.calls[0][0] as Request
    expect(new URL(request.url).pathname).toBe('/api/health')
  })

  it('reports when the backend is unreachable', async () => {
    stubFetch(new Response('down', { status: 502 }))

    render(<App />)

    expect(
      await screen.findByText(/Backend unreachable: Error: HTTP 502/),
    ).toBeInTheDocument()
  })
})
