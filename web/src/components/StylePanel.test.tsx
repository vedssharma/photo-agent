import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { makeDoc, stubApi } from '../test/fixtures'
import { StylePanel } from './StylePanel'

const learned = {
  tendencies: [],
  lines: ['warmer (temperature around +15, 3 edits)'],
  has_usual_look: true,
  signals: 3,
}
const blank = { tendencies: [], lines: [], has_usual_look: false, signals: 0 }

describe('StylePanel', () => {
  it('shows what it learned and applies the usual look', async () => {
    const applied = makeDoc({ head: 's1' })
    stubApi({
      'GET /api/style': () => Response.json(learned),
      'POST /api/documents/abc123abc123/style/usual-look': () =>
        Response.json(applied),
    })
    const onDocument = vi.fn()
    render(<StylePanel doc={makeDoc()} onDocument={onDocument} />)
    expect(
      await screen.findByText(
        'You like warmer (temperature around +15, 3 edits)',
      ),
    ).toBeVisible()
    await userEvent.click(screen.getByRole('button', { name: 'My usual look' }))
    expect(onDocument).toHaveBeenCalledWith(applied)
  })

  it('forgets on request', async () => {
    let forgotten = false
    stubApi({
      'GET /api/style': () => Response.json(forgotten ? blank : learned),
      'DELETE /api/style': () => {
        forgotten = true
        return new Response(null, { status: 204 })
      },
    })
    render(<StylePanel doc={makeDoc()} onDocument={vi.fn()} />)
    await userEvent.click(
      await screen.findByRole('button', { name: 'Forget my style' }),
    )
    expect(await screen.findByText(/Nothing learned yet/)).toBeVisible()
    expect(screen.getByRole('button', { name: 'My usual look' })).toBeDisabled()
  })
})
