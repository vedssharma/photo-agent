import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { makeDoc, makeStep, stubApi } from '../test/fixtures'
import { historyRows } from '../lib/history'
import { HistoryPanel } from './HistoryPanel'

const a = makeStep('a', { label: 'warmer' })
const b = makeStep('b', { parent: 'a', label: 'brighter', active: false })
const c = makeStep('c', { parent: 'a', label: 'cooler', kind: 'manual' })

describe('historyRows', () => {
  it('indents branches under the step they were made from', () => {
    expect(historyRows([a, b, c]).map((r) => [r.step.id, r.depth])).toEqual([
      ['a', 0],
      ['c', 1],
      ['b', 0],
    ])
  })
})

describe('HistoryPanel', () => {
  it('lists every step and marks the current one', () => {
    render(
      <HistoryPanel
        doc={makeDoc({ history: [a, b, c], head: 'c' })}
        onDocument={vi.fn()}
      />,
    )
    expect(screen.getByRole('button', { name: /cooler/ })).toHaveAttribute(
      'aria-current',
      'step',
    )
    expect(screen.getByRole('button', { name: /brighter/ })).toHaveClass(
      'inactive',
    )
    expect(screen.getByLabelText('Manual edit')).toBeInTheDocument()
  })

  it('jumps to a step when clicked', async () => {
    const jumped = makeDoc({ head: 'b' })
    const fetchMock = stubApi({
      'POST /api/documents/abc123abc123/checkout': () => Response.json(jumped),
    })
    const onDocument = vi.fn()
    render(
      <HistoryPanel
        doc={makeDoc({ history: [a, b, c], head: 'c' })}
        onDocument={onDocument}
      />,
    )
    await userEvent.click(screen.getByRole('button', { name: /brighter/ }))
    expect(onDocument).toHaveBeenCalledWith(jumped)
    const request = fetchMock.mock.calls[0][0]
    expect(await request.json()).toEqual({ step_id: 'b' })

    await userEvent.click(screen.getByRole('button', { name: /Original/ }))
    expect(await fetchMock.mock.calls[1][0].json()).toEqual({ step_id: null })
  })
})
