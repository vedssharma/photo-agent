import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import type { Mask } from '../api/documents'
import { DEFAULT_MASK_TOOL } from '../lib/masks'
import { MaskOverlay } from './MaskOverlay'

function place(el: HTMLElement) {
  el.getBoundingClientRect = () =>
    ({ left: 0, top: 0, width: 200, height: 100 }) as DOMRect
}

function setup(mask: Mask, tool = DEFAULT_MASK_TOOL) {
  const onChange = vi.fn()
  render(
    <MaskOverlay
      mask={mask}
      width={200}
      height={100}
      tool={tool}
      onChange={onChange}
    />,
  )
  const overlay = screen.getByTestId('mask-overlay')
  place(overlay)
  return { overlay, onChange }
}

const drag = (el: HTMLElement, points: [number, number][]) => {
  const [first, ...rest] = points
  fireEvent.pointerDown(el, { button: 0, clientX: first[0], clientY: first[1] })
  for (const [x, y] of rest)
    fireEvent.pointerMove(el, { clientX: x, clientY: y })
  const last = points[points.length - 1]
  fireEvent.pointerUp(el, { clientX: last[0], clientY: last[1] })
}

describe('MaskOverlay', () => {
  it('paints a brush stroke in photo fractions', () => {
    const { overlay, onChange } = setup(
      { kind: 'brush', strokes: [], invert: false },
      { ...DEFAULT_MASK_TOOL, size: 0.05, erase: true },
    )
    drag(overlay, [
      [20, 50],
      [100, 50],
      [180, 50],
    ])
    const [mask, label] = onChange.mock.calls[0]
    expect(label).toBe('Erase from mask')
    expect(mask.strokes).toEqual([
      {
        points: [
          [0.1, 0.5],
          [0.5, 0.5],
          [0.9, 0.5],
        ],
        size: 0.05,
        hardness: 50,
        erase: true,
      },
    ])
  })

  it('places a linear gradient by dragging', () => {
    const { overlay, onChange } = setup({
      kind: 'linear',
      start: [0.5, 0],
      end: [0.5, 0.5],
      invert: false,
    })
    drag(overlay, [
      [20, 10],
      [60, 30],
      [100, 90],
    ])
    expect(onChange).toHaveBeenCalledWith(
      { kind: 'linear', start: [0.1, 0.1], end: [0.5, 0.9], invert: false },
      'Place gradient',
    )
  })

  it('moves a radial gradient by its center', () => {
    const { overlay, onChange } = setup({
      kind: 'radial',
      center: [0.5, 0.5],
      radius_x: 0.2,
      radius_y: 0.3,
      feather: 50,
      invert: false,
    })
    drag(overlay, [
      [100, 50],
      [140, 60],
    ])
    const [mask, label] = onChange.mock.calls[0]
    expect(label).toBe('Move radial gradient')
    expect(mask.center[0]).toBeCloseTo(0.7)
    expect(mask.center[1]).toBeCloseTo(0.6)
    expect(mask.radius_x).toBe(0.2)
  })
})
