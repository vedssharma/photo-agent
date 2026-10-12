import { describe, expect, it } from 'vitest'

import { makePhoto } from '../test/fixtures'
import { coordinatesText, settingsText, shutterText, sizeText } from './details'

describe('photo details', () => {
  it('writes camera settings the way cameras do', () => {
    expect(shutterText(1 / 250)).toBe('1/250 s')
    expect(shutterText(2)).toBe('2 s')
    expect(
      settingsText(
        makePhoto('a', {
          f_number: 1.8,
          exposure_time: 0.004,
          iso: 125,
          focal_length: 6.86,
        }),
      ),
    ).toBe('ƒ/1.8 · 1/250 s · ISO 125 · 6.9 mm')
    expect(settingsText(makePhoto('a'))).toBeNull()
  })

  it('describes size and location', () => {
    expect(
      sizeText(makePhoto('a', { width: 4000, height: 3000, file_size: 3e6 })),
    ).toBe('4000 × 3000 (12.0 MP), 2.9 MB')
    expect(coordinatesText(-33.8568, 151.2153)).toBe('33.8568° S, 151.2153° E')
  })
})
