import { vi } from 'vitest'

import type { DocumentView } from '../api/documents'

export function makeDoc(overrides: Partial<DocumentView> = {}): DocumentView {
  return {
    id: 'abc123abc123',
    filename: 'cat.heic',
    format: 'HEIF',
    width: 4032,
    height: 3024,
    step: 0,
    revision: 'original',
    operations: [],
    can_undo: false,
    can_redo: false,
    turns: [],
    chat: [],
    ...overrides,
  }
}

type Route = (request: Request) => Response | Promise<Response>

/**
 * Stub `fetch` with handlers keyed by "METHOD /path". Unmatched requests fail the test
 * loudly with a 599 so they are easy to spot.
 */
export function stubApi(routes: Record<string, Route>) {
  const fetchMock = vi.fn(async (input: Request) => {
    const url = new URL(input.url)
    const handler = routes[`${input.method} ${url.pathname}`]
    if (!handler)
      return new Response(`no stub for ${input.method} ${url.pathname}`, {
        status: 599,
      })
    return handler(input)
  })
  vi.stubGlobal('fetch', fetchMock)
  return fetchMock
}

export const healthy: Route = () =>
  Response.json({ status: 'ok', version: '1.2.3', anthropic_configured: true })
