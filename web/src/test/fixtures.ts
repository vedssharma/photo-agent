import { vi } from 'vitest'

import type { DocumentView, StepView } from '../api/documents'
import type { OperationSpec } from '../api/operations'

export function makeDoc(overrides: Partial<DocumentView> = {}): DocumentView {
  return {
    id: 'abc123abc123',
    filename: 'cat.heic',
    format: 'HEIF',
    width: 4032,
    height: 3024,
    revision: 'original',
    state: { framing: [], layers: [] },
    head: null,
    tip: null,
    can_undo: false,
    can_redo: false,
    undo_label: null,
    redo_label: null,
    history: [],
    chat: [],
    ...overrides,
  }
}

export function makeStep(
  id: string,
  overrides: Partial<StepView> = {},
): StepView {
  return {
    id,
    parent: null,
    kind: 'agent',
    label: `step ${id}`,
    created_at: '2026-10-08T12:00:00Z',
    active: true,
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

const amount = (name: string, min = -100, max = 100, step = 1) => ({
  name,
  label: name.charAt(0).toUpperCase() + name.slice(1),
  kind: 'number' as const,
  description: '',
  min,
  max,
  step,
  choices: null,
  default: 0,
})

/** A few operation specs, shaped like GET /api/operations returns them. */
export const SPECS = new Map<string, OperationSpec>(
  (
    [
      {
        op: 'exposure',
        label: 'Exposure',
        description: '',
        group: 'light',
        framing: false,
        params: [amount('stops', -5, 5, 0.05)],
      },
      {
        op: 'contrast',
        label: 'Contrast',
        description: '',
        group: 'light',
        framing: false,
        params: [amount('amount')],
      },
      {
        op: 'white_balance',
        label: 'White balance',
        description: '',
        group: 'color',
        framing: false,
        params: [amount('temperature'), amount('tint')],
      },
      {
        op: 'crop',
        label: 'Crop',
        description: '',
        group: 'framing',
        framing: true,
        params: [
          { ...amount('left', 0, 1, 0.01) },
          { ...amount('top', 0, 1, 0.01) },
          { ...amount('right', 0, 1, 0.01), default: 1 },
          { ...amount('bottom', 0, 1, 0.01), default: 1 },
          {
            name: 'aspect',
            label: 'Aspect',
            kind: 'choice',
            description: '',
            min: null,
            max: null,
            step: null,
            choices: ['free', '1:1', '4:5'],
            default: 'free',
          },
        ],
      },
    ] satisfies OperationSpec[]
  ).map((s) => [s.op, s]),
)
