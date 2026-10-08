import { fireEvent, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { PhotoPicker } from './PhotoPicker'

function photo(name: string, type = 'image/jpeg') {
  return new File(['data'], name, { type })
}

describe('PhotoPicker', () => {
  it('opens a chosen photo', async () => {
    const onPick = vi.fn()
    render(<PhotoPicker onPick={onPick} />)

    await userEvent.upload(
      screen.getByLabelText('Photo file'),
      photo('cat.jpg'),
    )

    expect(onPick).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'cat.jpg' }),
    )
  })

  it('accepts HEIC files that have no MIME type', () => {
    const onPick = vi.fn()
    render(<PhotoPicker onPick={onPick} />)

    fireEvent.drop(window, {
      dataTransfer: { types: ['Files'], files: [photo('IMG_0001.HEIC', '')] },
    })

    expect(onPick).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'IMG_0001.HEIC' }),
    )
  })

  it('rejects files that are not photos', () => {
    const onPick = vi.fn()
    render(<PhotoPicker onPick={onPick} />)

    fireEvent.drop(window, {
      dataTransfer: {
        types: ['Files'],
        files: [photo('notes.pdf', 'application/pdf')],
      },
    })

    expect(onPick).not.toHaveBeenCalled()
    expect(screen.getByRole('alert')).toHaveTextContent(
      'notes.pdf is not a JPEG, PNG, or HEIC',
    )
  })

  it('highlights while a file is dragged over the page', () => {
    const { container } = render(<PhotoPicker onPick={vi.fn()} />)
    fireEvent.dragEnter(window, { dataTransfer: { types: ['Files'] } })
    expect(container.querySelector('.picker')).toHaveClass('dragging')
    fireEvent.dragLeave(window, { dataTransfer: { types: ['Files'] } })
    expect(container.querySelector('.picker')).not.toHaveClass('dragging')
  })
})
