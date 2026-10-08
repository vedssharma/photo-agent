import { act, renderHook } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { DocumentView, EditState } from '../api/documents'
import { makeDoc, stubApi } from '../test/fixtures'
import { useManualEdit } from './useManualEdit'

describe('useManualEdit', () => {
  it('applies queued changes on top of each other', async () => {
    const bodies: { label: string; state: EditState }[] = []
    stubApi({
      'POST /api/documents/abc123abc123/edits': async (req) => {
        const body = await req.json()
        bodies.push(body)
        return Response.json(makeDoc({ state: body.state }))
      },
    })
    const onDocument = vi.fn<(doc: DocumentView) => void>()
    const { result } = renderHook(() => useManualEdit(makeDoc(), onDocument))
    const withFraming = (n: number) => (s: EditState) => ({
      ...s,
      framing: [
        ...s.framing,
        { id: `r${n}`, op: 'rotate' as const, degrees: 90 as const },
      ],
    })

    await act(async () => {
      void result.current.edit('first', withFraming(1))
      await result.current.edit('second', withFraming(2))
    })

    expect(bodies.map((b) => b.state.framing.map((op) => op.id))).toEqual([
      ['r1'],
      ['r1', 'r2'],
    ])
    expect(onDocument).toHaveBeenCalledTimes(2)
    expect(result.current.working).toBe(false)
  })
})
