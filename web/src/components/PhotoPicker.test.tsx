import { fireEvent, render, screen, waitFor } from '@testing-library/react'
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

  it('opens a saved project from its own option', async () => {
    const onPick = vi.fn()
    render(<PhotoPicker onPick={onPick} />)

    expect(
      screen.getByRole('button', { name: 'Choose a photo' }),
    ).toBeInTheDocument()
    expect(
      screen.getByRole('button', { name: 'Open a project' }),
    ).toBeInTheDocument()
    await userEvent.upload(
      screen.getByLabelText('Project file'),
      new File(['zip'], 'trip.photoagent'),
    )

    expect(onPick).toHaveBeenCalledWith(
      expect.objectContaining({ name: 'trip.photoagent' }),
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
      'notes.pdf is not a JPEG, PNG, HEIC, TIFF, or RAW',
    )
  })

  it('highlights while a file is dragged over the page', () => {
    const { container } = render(<PhotoPicker onPick={vi.fn()} />)
    fireEvent.dragEnter(window, { dataTransfer: { types: ['Files'] } })
    expect(container.querySelector('.picker')).toHaveClass('dragging')
    fireEvent.dragLeave(window, { dataTransfer: { types: ['Files'] } })
    expect(container.querySelector('.picker')).not.toHaveClass('dragging')
  })

  it('imports several dropped photos into the library', async () => {
    const onPick = vi.fn()
    const onImport = vi.fn()
    render(<PhotoPicker onPick={onPick} onImport={onImport} />)

    const files = [photo('a.jpg'), photo('b.jpg')]
    fireEvent.drop(window, {
      dataTransfer: { types: ['Files'], files, items: [] },
    })

    await waitFor(() => expect(onImport).toHaveBeenCalledWith(files))
    expect(onPick).not.toHaveBeenCalled()
  })

  it('offers to import a folder only where there is a library', async () => {
    const onImport = vi.fn()
    const { rerender } = render(<PhotoPicker onPick={vi.fn()} />)
    expect(screen.queryByRole('button', { name: 'Import a folder' })).toBeNull()

    rerender(<PhotoPicker onPick={vi.fn()} onImport={onImport} />)
    await userEvent.upload(screen.getByLabelText('Folder to import'), [
      photo('1.jpg'),
      photo('2.jpg'),
    ])
    expect(onImport).toHaveBeenCalledWith([
      expect.objectContaining({ name: '1.jpg' }),
      expect.objectContaining({ name: '2.jpg' }),
    ])
  })
})
