import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { dispositionFilename } from '../api/documents'
import { makeDoc, stubApi } from '../test/fixtures'
import { ExportDialog } from './ExportDialog'

describe('ExportDialog', () => {
  it('downloads a JPEG with the chosen quality and no location by default', async () => {
    const fetchMock = stubApi({
      'POST /api/documents/abc123abc123/export': () =>
        new Response('jpg', {
          headers: {
            'content-type': 'image/jpeg',
            'content-disposition':
              "attachment; filename*=UTF-8''cat-edited.jpg",
          },
        }),
    })
    const save = vi.fn()
    const onClose = vi.fn()
    render(<ExportDialog doc={makeDoc()} onClose={onClose} save={save} />)

    await userEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() => expect(save).toHaveBeenCalled())
    expect(save.mock.calls[0][0].filename).toBe('cat-edited.jpg')
    const request = fetchMock.mock.calls[0][0]
    expect(await request.json()).toEqual({
      format: 'jpeg',
      quality: 92,
      keep_location: false,
    })
    expect(onClose).toHaveBeenCalled()
  })

  it('exports PNG without a quality setting, keeping location when asked', async () => {
    const fetchMock = stubApi({
      'POST /api/documents/abc123abc123/export': () => new Response('png'),
    })
    const save = vi.fn()
    render(<ExportDialog doc={makeDoc()} onClose={vi.fn()} save={save} />)

    await userEvent.click(screen.getByRole('radio', { name: /PNG/ }))
    expect(screen.queryByRole('slider')).not.toBeInTheDocument()
    await userEvent.click(
      screen.getByRole('checkbox', { name: /Keep GPS location/ }),
    )
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() => expect(save).toHaveBeenCalled())
    expect(save.mock.calls[0][0].filename).toBe('photo.png')
    expect(await fetchMock.mock.calls[0][0].json()).toMatchObject({
      format: 'png',
      keep_location: true,
    })
  })

  it('reports failures and stays open', async () => {
    stubApi({
      'POST /api/documents/abc123abc123/export': () =>
        Response.json({ detail: 'No such document.' }, { status: 404 }),
    })
    const onClose = vi.fn()
    render(<ExportDialog doc={makeDoc()} onClose={onClose} save={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))
    expect(await screen.findByRole('alert')).toHaveTextContent(
      'Export failed: No such document.',
    )
    expect(onClose).not.toHaveBeenCalled()
  })

  it('closes with Escape', async () => {
    const onClose = vi.fn()
    render(<ExportDialog doc={makeDoc()} onClose={onClose} save={vi.fn()} />)
    await userEvent.keyboard('{Escape}')
    expect(onClose).toHaveBeenCalled()
  })
})

describe('dispositionFilename', () => {
  it('prefers the UTF-8 name and falls back', () => {
    expect(
      dispositionFilename("attachment; filename*=UTF-8''caf%C3%A9.jpg", 'x'),
    ).toBe('café.jpg')
    expect(dispositionFilename('attachment; filename="a.png"', 'x')).toBe(
      'a.png',
    )
    expect(dispositionFilename(null, 'x.jpg')).toBe('x.jpg')
  })
})
