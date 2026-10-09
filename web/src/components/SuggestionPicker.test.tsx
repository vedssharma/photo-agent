import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { makeDoc, stubApi } from '../test/fixtures'
import { SuggestionPicker } from './SuggestionPicker'

const set = {
  revision: 'original',
  source: 'claude',
  suggestions: [
    {
      id: 's1',
      title: 'Golden hour',
      description: 'Warm, low sun.',
      layers: [],
    },
    { id: 's2', title: 'Moody', description: 'Dark and rich.', layers: [] },
  ],
}

describe('SuggestionPicker', () => {
  it('shows thumbnails and applies the one picked', async () => {
    const applied = makeDoc({ head: 'x1', revision: 'r1' })
    const fetchMock = stubApi({
      'POST /api/documents/abc123abc123/suggestions': () => Response.json(set),
      'POST /api/documents/abc123abc123/suggestions/s1': () =>
        Response.json(applied),
    })
    const onDocument = vi.fn()
    render(<SuggestionPicker doc={makeDoc()} onDocument={onDocument} />)

    expect(screen.getByText('Looking for ideas for this photo…')).toBeVisible()
    const golden = await screen.findByRole('button', { name: /Golden hour/ })
    expect(golden.querySelector('img')).toHaveAttribute(
      'src',
      '/api/documents/abc123abc123/suggestions/s1/preview?revision=original',
    )
    await userEvent.click(golden)
    expect(onDocument).toHaveBeenCalledWith(applied)
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it('says so quietly when there are none', async () => {
    stubApi({
      'POST /api/documents/abc123abc123/suggestions': () =>
        Response.json({ detail: 'Claude is busy.' }, { status: 502 }),
    })
    render(<SuggestionPicker doc={makeDoc()} onDocument={vi.fn()} />)
    expect(
      await screen.findByText('No suggestions right now (Claude is busy.).'),
    ).toBeVisible()
  })
})
