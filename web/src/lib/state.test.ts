import { describe, expect, it } from 'vitest'

import type { EditState } from '../api/documents'
import { removeOperation, setCutout, updateOperation } from './state'

const state: EditState = {
  framing: [
    {
      id: 'c1',
      op: 'crop',
      left: 0,
      top: 0,
      right: 1,
      bottom: 1,
      aspect: '1:1',
    },
  ],
  layers: [],
  cutout: {
    visible: true,
    background: null,
    mask: {
      kind: 'semantic',
      target: 'subject',
      points: [],
      strokes: [],
      description: '',
      invert: false,
    },
  },
}

describe('edit state helpers', () => {
  it('keep the cutout when changing operations', () => {
    expect(updateOperation(state, 'c1', { aspect: '4:5' }).cutout).toBe(
      state.cutout,
    )
    expect(removeOperation(state, 'c1').cutout).toBe(state.cutout)
  })

  it('set, change, and remove the cutout', () => {
    const white = setCutout(state, (c) => ({ ...c, background: '#ffffff' }))
    expect(white.cutout?.background).toBe('#ffffff')
    expect(setCutout(state, null).cutout).toBeNull()
    const none = { ...state, cutout: null }
    expect(setCutout(none, (c) => ({ ...c, visible: false }))).toBe(none)
  })
})
