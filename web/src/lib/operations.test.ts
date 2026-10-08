import { describe, expect, it } from 'vitest'

import { opSummary } from './operations'

describe('opSummary', () => {
  it('reads like the server summary', () => {
    expect(opSummary({ id: 'a', op: 'exposure', stops: 0.4 })).toBe(
      'Exposure (stops +0.4)',
    )
    expect(
      opSummary({ id: 'b', op: 'white_balance', temperature: -20, tint: 0 }),
    ).toBe('White balance (temperature -20, tint 0)')
  })
})
