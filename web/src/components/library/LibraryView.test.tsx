import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { useImporter } from '../../hooks/useImporter'
import { idle, makePhoto, stubApi } from '../../test/fixtures'
import { LibraryView } from './LibraryView'

function Harness({ onEdit = vi.fn() }: { onEdit?: (id: string) => void }) {
  const importer = useImporter()
  return <LibraryView importer={importer} onEdit={onEdit} pollMs={10} />
}

const PHOTOS = [
  makePhoto('aaaaaaaaaaaa'),
  makePhoto('bbbbbbbbbbbb'),
  makePhoto('cccccccccccc'),
]

describe('LibraryView', () => {
  it('shows every photo and opens the selected one in the editor', async () => {
    stubApi({
      'GET /api/library/photos': () => Response.json(PHOTOS),
      'GET /api/library/status': idle,
    })
    const onEdit = vi.fn()
    render(<Harness onEdit={onEdit} />)

    const options = await screen.findAllByRole('option')
    expect(options).toHaveLength(3)
    expect(screen.getByText('3 photos')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Edit' })).toBeDisabled()

    await userEvent.click(options[1])
    expect(options[1]).toHaveAttribute('aria-selected', 'true')
    await userEvent.click(screen.getByRole('button', { name: 'Edit' }))
    expect(onEdit).toHaveBeenCalledWith('bbbbbbbbbbbb')
  })

  it('looks at one photo large with a filmstrip to move through', async () => {
    stubApi({
      'GET /api/library/photos': () => Response.json(PHOTOS),
      'GET /api/library/status': idle,
    })
    render(<Harness />)

    const options = await screen.findAllByRole('option')
    fireEvent.doubleClick(options[0])

    const large = screen.getByRole('img', { name: 'aaaaaaaaaaaa.jpg' })
    expect(large).toHaveAttribute(
      'src',
      '/api/documents/aaaaaaaaaaaa/preview?revision=original',
    )
    expect(
      screen.getByRole('listbox', { name: 'Filmstrip' }),
    ).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(
      screen.getByRole('img', { name: 'bbbbbbbbbbbb.jpg' }),
    ).toBeInTheDocument()

    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.getByRole('listbox', { name: 'Photos' })).toBeInTheDocument()
  })

  it('imports picked photos and refreshes the grid', async () => {
    let listed = [PHOTOS[0]]
    const fetchMock = stubApi({
      'GET /api/library/photos': () => Response.json(listed),
      'GET /api/library/status': idle,
      'POST /api/library/photos': () => {
        listed = PHOTOS
        return Response.json(
          {
            imported: [PHOTOS[1], PHOTOS[2]],
            skipped: [
              {
                filename: 'a.jpg',
                reason: 'Already',
                existing: 'aaaaaaaaaaaa',
              },
            ],
          },
          { status: 201 },
        )
      },
    })
    render(<Harness />)
    expect(await screen.findAllByRole('option')).toHaveLength(1)

    await userEvent.upload(screen.getByLabelText('Photos to import'), [
      new File(['1'], 'a.jpg', { type: 'image/jpeg' }),
      new File(['2'], 'b.jpg', { type: 'image/jpeg' }),
      new File(['3'], 'c.png', { type: 'image/png' }),
    ])

    expect(
      await screen.findByText(
        /Imported 2 of 3 photos\. 1 already in the library\./,
      ),
    ).toBeInTheDocument()
    await waitFor(() => expect(screen.getAllByRole('option')).toHaveLength(3))
    const post = fetchMock.mock.calls
      .map(([r]) => r as Request)
      .find((r) => r.method === 'POST')!
    expect((await post.formData()).getAll('files')).toHaveLength(3)
  })

  it('reports indexing while it runs', async () => {
    let pending = 2
    stubApi({
      'GET /api/library/photos': () => Response.json(PHOTOS),
      'GET /api/library/status': () => {
        const body = { total: 3, pending, current: 'Making a thumbnail: c.jpg' }
        pending = 0
        return Response.json(body)
      },
    })
    render(<Harness />)
    expect(
      await screen.findByText(
        /Indexing 2 of 3 photos \(Making a thumbnail: c\.jpg\)/,
      ),
    ).toBeInTheDocument()
    await waitFor(() =>
      expect(screen.queryByText(/Indexing/)).not.toBeInTheDocument(),
    )
  })

  it('deletes the selection after asking', async () => {
    let listed = PHOTOS
    const removed = vi.fn()
    stubApi({
      'GET /api/library/photos': () => Response.json(listed),
      'GET /api/library/status': idle,
      'POST /api/library/photos/remove': async (req) => {
        const { ids } = (await req.json()) as { ids: string[] }
        removed(ids)
        listed = PHOTOS.filter((p) => !ids.includes(p.id))
        return Response.json({ removed: ids.length })
      },
    })
    vi.stubGlobal(
      'confirm',
      vi.fn(() => true),
    )
    render(<Harness />)
    const options = await screen.findAllByRole('option')
    await userEvent.click(options[0])
    fireEvent.click(options[2], { shiftKey: true })
    expect(screen.getByText(/3 selected/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }))
    await waitFor(() =>
      expect(removed).toHaveBeenCalledWith(PHOTOS.map((p) => p.id)),
    )
    await waitFor(() => expect(screen.queryAllByRole('option')).toHaveLength(0))
    expect(screen.getByText('Your library is empty')).toBeInTheDocument()
  })
})
