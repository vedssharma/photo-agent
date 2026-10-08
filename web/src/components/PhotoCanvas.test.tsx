import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { PhotoCanvas } from './PhotoCanvas'

function setup() {
  render(<PhotoCanvas src="/after.jpg" beforeSrc="/before.jpg" alt="cat" />)
  return { photo: () => screen.getByRole('img', { name: /^cat/ }) }
}

describe('PhotoCanvas', () => {
  it('shows the edited photo fitted to the window', () => {
    const { photo } = setup()
    expect(photo()).toHaveAttribute('src', '/after.jpg')
    expect(
      screen.getByRole('button', { name: 'Fit to window' }),
    ).toHaveTextContent('100%')
  })

  it('zooms with the buttons and the wheel, and fits again', () => {
    setup()
    const fit = screen.getByRole('button', { name: 'Fit to window' })
    fireEvent.click(screen.getByRole('button', { name: '+' }))
    expect(fit).toHaveTextContent('150%')
    fireEvent.wheel(screen.getByRole('img', { name: 'cat' }), { deltaY: -200 })
    expect(fit).not.toHaveTextContent('150%')
    fireEvent.click(fit)
    expect(fit).toHaveTextContent('100%')
    expect(screen.getByRole('button', { name: '−' })).toBeDisabled()
  })

  it('pans by dragging when zoomed in', () => {
    const { container } = render(
      <PhotoCanvas src="/a.jpg" beforeSrc="/b.jpg" alt="dog" />,
    )
    fireEvent.click(screen.getByRole('button', { name: '+' }))
    const viewport = container.querySelector('.viewport')!
    fireEvent.pointerDown(viewport, { clientX: 10, clientY: 10, button: 0 })
    fireEvent.pointerMove(viewport, { clientX: 40, clientY: 30 })
    fireEvent.pointerUp(viewport)
    const stage = container.querySelector<HTMLElement>('.stage')!
    expect(stage.style.transform).toBe('translate(30px, 20px) scale(1.5)')
  })

  it('shows the original while Compare is held', () => {
    const { photo } = setup()
    const compare = screen.getByRole('button', { name: 'Hold to compare' })
    fireEvent.pointerDown(compare)
    expect(photo()).toHaveAttribute('src', '/before.jpg')
    expect(screen.getByText('Original')).toBeInTheDocument()
    fireEvent.pointerUp(compare)
    expect(photo()).toHaveAttribute('src', '/after.jpg')
  })

  it('compares with the backslash key unless typing', () => {
    const { photo } = setup()
    fireEvent.keyDown(window, { key: '\\' })
    expect(photo()).toHaveAttribute('src', '/before.jpg')
    fireEvent.keyUp(window, { key: '\\' })
    expect(photo()).toHaveAttribute('src', '/after.jpg')
  })

  it('splits before and after with a slider', () => {
    const { container } = render(
      <PhotoCanvas src="/a.jpg" beforeSrc="/b.jpg" alt="dog" />,
    )
    fireEvent.click(screen.getByRole('button', { name: 'Split view' }))
    const slider = screen.getByRole('slider', {
      name: 'Before and after split',
    })
    fireEvent.change(slider, { target: { value: '30' } })
    const before = container.querySelector<HTMLElement>('.photo.before')!
    expect(before).toHaveAttribute('src', '/b.jpg')
    expect(before.style.clipPath).toBe('inset(0 70% 0 0)')
  })
})
