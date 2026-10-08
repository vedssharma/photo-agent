import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { Slider } from './Slider'

describe('Slider', () => {
  it('reports movement live and commits once when released', () => {
    const onInput = vi.fn()
    const onCommit = vi.fn()
    render(
      <Slider
        label="Opacity"
        value={100}
        min={0}
        max={100}
        format={(v) => `${v}%`}
        onInput={onInput}
        onCommit={onCommit}
      />,
    )
    const slider = screen.getByRole('slider', { name: 'Opacity' })
    fireEvent.change(slider, { target: { value: '60' } })
    fireEvent.change(slider, { target: { value: '40' } })
    expect(onInput).toHaveBeenLastCalledWith(40)
    expect(screen.getByText('40%')).toBeInTheDocument()
    expect(onCommit).not.toHaveBeenCalled()

    fireEvent.pointerUp(slider)
    expect(onCommit).toHaveBeenCalledExactlyOnceWith(40)
    fireEvent.blur(slider)
    expect(onCommit).toHaveBeenCalledOnce()
  })
})
