import { describe, expect, it } from 'vitest'

import { IMPORT_BATCH_FILES, importBatches, photoFiles } from './library'

const file = (name: string, size = 10) =>
  new File([new Uint8Array(size)], name, { type: '' })

describe('library imports', () => {
  it('sends big imports in several requests', () => {
    const many = Array.from({ length: IMPORT_BATCH_FILES * 2 + 1 }, (_, i) =>
      file(`${i}.jpg`),
    )
    expect(importBatches(many).map((b) => b.length)).toEqual([
      IMPORT_BATCH_FILES,
      IMPORT_BATCH_FILES,
      1,
    ])
    const big = [file('a.jpg', 30e6), file('b.jpg', 30e6), file('c.jpg', 1)]
    expect(importBatches(big).map((b) => b.length)).toEqual([1, 2])
  })

  it('keeps only photos from a folder', () => {
    const found = photoFiles([
      file('IMG_1.HEIC'),
      file('notes.txt'),
      file('.DS_Store'),
      file('trip.photoagent'),
      file('b.png'),
    ])
    expect(found.map((f) => f.name)).toEqual(['IMG_1.HEIC', 'b.png'])
  })
})
