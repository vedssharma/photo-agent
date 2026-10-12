import { describe, expect, it } from 'vitest'

import {
  NOTHING,
  clickSelect,
  keepOnly,
  moveFocus,
  selectAll,
  selectedIds,
} from './selection'

const ORDER = ['a', 'b', 'c', 'd', 'e']

describe('selection', () => {
  it('selects one photo per plain click', () => {
    const sel = clickSelect(clickSelect(NOTHING, ORDER, 'b'), ORDER, 'd')
    expect([...sel.ids]).toEqual(['d'])
    expect(sel.focus).toBe('d')
  })

  it('extends a range with shift and toggles with ctrl or cmd', () => {
    let sel = clickSelect(NOTHING, ORDER, 'b')
    sel = clickSelect(sel, ORDER, 'd', { range: true })
    expect(selectedIds(sel, ORDER)).toEqual(['b', 'c', 'd'])
    sel = clickSelect(sel, ORDER, 'c', { toggle: true })
    expect(selectedIds(sel, ORDER)).toEqual(['b', 'd'])
    sel = clickSelect(sel, ORDER, 'e', { toggle: true })
    expect(selectedIds(sel, ORDER)).toEqual(['b', 'd', 'e'])
  })

  it('moves the focus by places and rows, staying in bounds', () => {
    let sel = moveFocus(NOTHING, ORDER, 1)
    expect(sel.focus).toBe('a')
    sel = moveFocus(sel, ORDER, 3)
    expect(sel.focus).toBe('d')
    sel = moveFocus(sel, ORDER, 10)
    expect(sel.focus).toBe('e')
    sel = moveFocus(sel, ORDER, -2, true)
    expect(selectedIds(sel, ORDER)).toEqual(['c', 'd', 'e'])
  })

  it('drops photos that are gone', () => {
    const sel = selectAll(ORDER, clickSelect(NOTHING, ORDER, 'c'))
    const kept = keepOnly(sel, ['a', 'b'])
    expect(selectedIds(kept, ORDER)).toEqual(['a', 'b'])
    expect(kept.focus).toBeNull()
    expect(keepOnly(kept, ['a', 'b'])).toBe(kept)
  })
})
