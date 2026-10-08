import { describe, expect, it } from 'vitest'

import type { EditState } from '../api/documents'
import {
  NEUTRAL,
  adjustPixel,
  deltaAdjustment,
  liveTarget,
} from './livePreview'

describe('deltaAdjustment', () => {
  it('turns a slider move into the change to apply', () => {
    expect(
      deltaAdjustment(
        { id: 'e', op: 'exposure', stops: 0.5 },
        { id: 'e', op: 'exposure', stops: 1 },
      ),
    ).toEqual({ ...NEUTRAL, exposure: 0.5 })
    expect(
      deltaAdjustment(
        { id: 'c', op: 'contrast', amount: 10 },
        { id: 'c', op: 'contrast', amount: 30 },
      )?.contrast,
    ).toBeCloseTo(0.2)
    expect(
      deltaAdjustment(
        { id: 's', op: 'saturation', amount: -50 },
        { id: 's', op: 'saturation', amount: 0 },
      )?.saturation,
    ).toBeCloseTo(2)
  })

  it('skips operations it cannot preview', () => {
    expect(
      deltaAdjustment(
        { id: 'x', op: 'clarity', amount: 0 },
        { id: 'x', op: 'clarity', amount: 50 },
      ),
    ).toBeNull()
  })
})

describe('adjustPixel', () => {
  it('changes nothing when neutral or fully masked', () => {
    const px: [number, number, number] = [0.2, 0.5, 0.7]
    adjustPixel(px, NEUTRAL).forEach((v, i) => expect(v).toBeCloseTo(px[i], 5))
    const bright = { ...NEUTRAL, exposure: 1 }
    adjustPixel(px, bright, 0).forEach((v, i) =>
      expect(v).toBeCloseTo(px[i], 5),
    )
  })

  it('doubles linear light for +1 stop, like the server', () => {
    const [r] = adjustPixel([0.5, 0.5, 0.5], { ...NEUTRAL, exposure: 1 })
    // 0.5 sRGB is 0.214 linear; doubled is 0.428, which is 0.686 sRGB.
    expect(r).toBeCloseTo(0.686, 2)
  })

  it('desaturates with saturation 0', () => {
    const [r, g, b] = adjustPixel([0.8, 0.4, 0.2], {
      ...NEUTRAL,
      saturation: 0,
    })
    expect(r).toBeCloseTo(g, 5)
    expect(g).toBeCloseTo(b, 5)
  })
})

describe('liveTarget', () => {
  const state: EditState = {
    framing: [{ id: 'r', op: 'rotate', degrees: 90 }],
    layers: [
      {
        id: 'L1',
        name: 'Sky',
        visible: true,
        opacity: 50,
        blend_mode: 'normal',
        mask: {
          kind: 'linear',
          start: [0.5, 0],
          end: [0.5, 0.5],
          invert: false,
        },
        operations: [{ id: 'e', op: 'exposure', stops: 0 }],
      },
    ],
  }

  it('uses the layer opacity and mask', () => {
    expect(
      liveTarget(state, { layerId: 'L1', opId: 'e', param: 'stops', value: 1 }),
    ).toEqual({
      adjustment: { ...NEUTRAL, exposure: 1 },
      opacity: 0.5,
      maskLayerId: 'L1',
    })
  })

  it('does not preview framing or hidden layers', () => {
    expect(
      liveTarget(state, {
        layerId: null,
        opId: 'r',
        param: 'degrees',
        value: 180,
      }),
    ).toBeNull()
    const hidden = {
      ...state,
      layers: [{ ...state.layers[0], visible: false }],
    }
    expect(
      liveTarget(hidden, {
        layerId: 'L1',
        opId: 'e',
        param: 'stops',
        value: 1,
      }),
    ).toBeNull()
  })
})
