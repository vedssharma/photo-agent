import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { FilterBar } from './FilterBar'

const FACETS = {
  cameras: [{ value: 'FUJIFILM X-T5', label: 'FUJIFILM X-T5', count: 3 }],
  places: [{ value: '36.95,-122.05', label: '37.0° N, 122.0° W', count: 2 }],
  years: [
    { value: '2025', label: '2025', count: 4 },
    { value: '2024', label: '2024', count: 1 },
  ],
}

describe('FilterBar', () => {
  it('filters by a whole year, a camera, and a place', async () => {
    const onFilter = vi.fn()
    const { rerender } = render(
      <FilterBar filter={{}} onFilter={onFilter} facets={FACETS} />,
    )
    await userEvent.selectOptions(screen.getByLabelText('Year'), '2025')
    expect(onFilter).toHaveBeenLastCalledWith({
      taken_after: '2025-01-01',
      taken_before: '2025-12-31',
    })

    rerender(
      <FilterBar
        filter={{ taken_after: '2025-01-01', taken_before: '2025-12-31' }}
        onFilter={onFilter}
        facets={FACETS}
      />,
    )
    expect(screen.getByLabelText('Year')).toHaveValue('2025')
    await userEvent.selectOptions(
      screen.getByLabelText('Camera'),
      'FUJIFILM X-T5',
    )
    expect(onFilter).toHaveBeenLastCalledWith(
      expect.objectContaining({ camera: 'FUJIFILM X-T5' }),
    )
    await userEvent.selectOptions(
      screen.getByLabelText('Place'),
      'Near 37.0° N, 122.0° W (2)',
    )
    expect(onFilter).toHaveBeenLastCalledWith(
      expect.objectContaining({ place: '36.95,-122.05' }),
    )
  })

  it('hides facets the library has nothing for', () => {
    render(
      <FilterBar
        filter={{}}
        onFilter={vi.fn()}
        facets={{ cameras: [], places: [], years: [] }}
      />,
    )
    expect(screen.queryByLabelText('Camera')).toBeNull()
    expect(screen.queryByLabelText('Place')).toBeNull()
    expect(screen.getByLabelText('From')).toBeInTheDocument()
  })
})
