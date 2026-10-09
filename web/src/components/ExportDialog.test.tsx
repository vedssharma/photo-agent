import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { dispositionFilename } from '../api/documents'
import { makeDoc, stubApi } from '../test/fixtures'
import { ExportDialog } from './ExportDialog'

const SUBJECT = {
  kind: 'semantic' as const,
  target: 'subject' as const,
  points: [],
  strokes: [],
  description: '',
  invert: false,
}

describe('ExportDialog', () => {
  it('picks PNG for a transparent cutout and warns about JPEG', async () => {
    const doc = makeDoc({
      state: {
        framing: [],
        layers: [],
        cutout: { visible: true, background: null, mask: SUBJECT },
      },
    })
    render(<ExportDialog doc={doc} onClose={vi.fn()} save={vi.fn()} />)
    expect(screen.getByRole('radio', { name: /PNG/ })).toBeChecked()
    expect(screen.getByText(/keeps the transparent background/)).toBeVisible()
    await userEvent.click(screen.getByRole('radio', { name: /JPEG/ }))
    expect(screen.getByText(/background will be white/)).toBeVisible()
  })

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
      upscale: 1,
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
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: /Enlarge/ }),
      '2',
    )
    expect(screen.getByText(/Good for prints/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))

    await waitFor(() => expect(save).toHaveBeenCalled())
    expect(save.mock.calls[0][0].filename).toBe('photo.png')
    expect(await fetchMock.mock.calls[0][0].json()).toMatchObject({
      format: 'png',
      keep_location: true,
      upscale: 2,
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
