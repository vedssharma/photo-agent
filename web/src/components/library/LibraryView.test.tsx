import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { useImporter } from '../../hooks/useImporter'
import { libraryRoutes, makePhoto, stubApi } from '../../test/fixtures'
import { LibraryView } from './LibraryView'

function Harness({ onEdit = vi.fn() }: { onEdit?: (id: string) => void }) {
  const importer = useImporter()
  return <LibraryView importer={importer} onEdit={onEdit} pollMs={10} />
}

const grid = () => screen.getByRole('listbox', { name: 'Photos' })
const thumbs = async () =>
  within(await screen.findByRole('listbox', { name: 'Photos' })).findAllByRole(
    'option',
  )

const PHOTOS = [
  makePhoto('aaaaaaaaaaaa'),
  makePhoto('bbbbbbbbbbbb'),
  makePhoto('cccccccccccc'),
]

describe('LibraryView', () => {
  it('shows every photo and opens the selected one in the editor', async () => {
    stubApi({
      ...libraryRoutes(() => PHOTOS),
    })
    const onEdit = vi.fn()
    render(<Harness onEdit={onEdit} />)

    const options = await thumbs()
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
      ...libraryRoutes(() => PHOTOS),
    })
    render(<Harness />)

    const options = await thumbs()
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
      ...libraryRoutes(() => listed),
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
    expect(await thumbs()).toHaveLength(1)

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
    await waitFor(() =>
      expect(within(grid()).getAllByRole('option')).toHaveLength(3),
    )
    const post = fetchMock.mock.calls
      .map(([r]) => r as Request)
      .find((r) => r.method === 'POST')!
    expect((await post.formData()).getAll('files')).toHaveLength(3)
  })

  it('reports indexing while it runs', async () => {
    let pending = 2
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/library/status': () =>
        Response.json({
          total: 3,
          pending,
          current: 'Making a thumbnail: c.jpg',
        }),
    })
    render(<Harness />)
    expect(
      await screen.findByText(
        /Indexing 2 of 3 photos \(Making a thumbnail: c\.jpg\)/,
      ),
    ).toBeInTheDocument()
    pending = 0
    await waitFor(() =>
      expect(screen.queryByText(/Indexing/)).not.toBeInTheDocument(),
    )
  })

  it('deletes the selection after asking', async () => {
    let listed = PHOTOS
    const removed = vi.fn()
    stubApi({
      ...libraryRoutes(() => listed),
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
    const options = await thumbs()
    await userEvent.click(options[0])
    fireEvent.click(options[2], { shiftKey: true })
    expect(screen.getByText(/3 selected/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Delete' }))
    await waitFor(() =>
      expect(removed).toHaveBeenCalledWith(PHOTOS.map((p) => p.id)),
    )
    await waitFor(() =>
      expect(screen.queryAllByRole('option', { name: /\.jpg$/ })).toHaveLength(
        0,
      ),
    )
    expect(screen.getByText('Your library is empty')).toBeInTheDocument()
  })

  it('rates, flags, and labels the selection from keys and the info panel', async () => {
    let listed = PHOTOS
    const marks: unknown[] = []
    stubApi({
      ...libraryRoutes(() => listed),
      'POST /api/library/photos/marks': async (req) => {
        const body = (await req.json()) as { ids: string[]; rating?: number }
        marks.push(body)
        listed = PHOTOS.map((p) =>
          body.ids.includes(p.id)
            ? { ...p, rating: body.rating ?? p.rating, flag: 'picked' as const }
            : p,
        )
        return Response.json({ changed: body.ids.length, photos: [] })
      },
    })
    render(<Harness />)
    const options = await thumbs()
    await userEvent.click(options[0])
    fireEvent.click(options[1], { shiftKey: true })

    fireEvent.keyDown(window, { key: '4' })
    await waitFor(() =>
      expect(marks).toEqual([
        { ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb'], rating: 4 },
      ]),
    )
    await waitFor(() =>
      expect(within(grid()).getAllByLabelText('4 stars')).toHaveLength(2),
    )
    expect(within(grid()).getAllByLabelText('Picked')).toHaveLength(2)

    const info = screen.getByRole('complementary', { name: 'Photo info' })
    expect(within(info).getByText('2 photos selected')).toBeInTheDocument()
    await userEvent.click(within(info).getByRole('button', { name: 'Reject' }))
    await userEvent.click(
      within(info).getByRole('button', { name: 'Green label' }),
    )
    await waitFor(() => expect(marks).toHaveLength(3))
    expect(marks.slice(1)).toEqual([
      { ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb'], flag: 'rejected' },
      { ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb'], label: 'green' },
    ])
  })

  it('makes albums from the selection and shows one album at a time', async () => {
    const album = {
      id: 'album1',
      name: 'Beach',
      created_at: '2026-10-12T12:00:00Z',
      count: 1,
      cover: 'cccccccccccc',
    }
    let albums: (typeof album)[] = []
    const created = vi.fn()
    const fetchMock = stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/library/photos': (req) =>
        Response.json(
          new URL(req.url).searchParams.get('album') === 'album1'
            ? [PHOTOS[2]]
            : PHOTOS,
        ),
      'GET /api/library/albums': () => Response.json(albums),
      'POST /api/library/albums': async (req) => {
        created(await req.json())
        albums = [album]
        return Response.json(album, { status: 201 })
      },
    })
    vi.stubGlobal(
      'prompt',
      vi.fn(() => ' Beach '),
    )
    render(<Harness />)
    const options = await thumbs()
    await userEvent.click(options[2])
    await userEvent.click(screen.getByRole('button', { name: 'New' }))

    expect(created).toHaveBeenCalledWith({
      name: 'Beach',
      photo_ids: ['cccccccccccc'],
    })
    await userEvent.click(await screen.findByRole('button', { name: /Beach/ }))
    await waitFor(() =>
      expect(within(grid()).getAllByRole('option')).toHaveLength(1),
    )
    const urls = fetchMock.mock.calls.map(([r]) => (r as Request).url)
    expect(
      urls.some((u) => u.endsWith('/api/library/photos?album=album1')),
    ).toBe(true)
    // An empty filter sends no query at all.
    expect(urls.some((u) => u.endsWith('/api/library/photos'))).toBe(true)
  })

  it('searches in everyday words and says how it understood them', async () => {
    const searched = vi.fn()
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'POST /api/library/search': async (req) => {
        searched(await req.json())
        return Response.json({
          query: 'beach last summer',
          understood: {
            tags: ['beach'],
            taken_after: '2026-06-01',
            taken_before: '2026-08-31',
            camera: null,
            min_rating: 0,
            flag: null,
            best: null,
            words: '',
          },
          summary: 'Tagged beach, taken Jun 1, 2026 to Aug 31, 2026',
          source: 'claude',
          matched_words_by: null,
          photos: [PHOTOS[2]],
        })
      },
    })
    render(<Harness />)
    await thumbs()
    await userEvent.type(
      screen.getByLabelText('Search photos'),
      'beach last summer{Enter}',
    )

    expect(searched).toHaveBeenCalledWith({
      query: 'beach last summer',
      album: null,
    })
    expect(
      await screen.findByText(
        /Tagged beach, taken Jun 1, 2026 to Aug 31, 2026/,
      ),
    ).toBeInTheDocument()
    expect(within(grid()).getAllByRole('option')).toHaveLength(1)

    await userEvent.click(screen.getByRole('button', { name: 'Clear search' }))
    expect(within(grid()).getAllByRole('option')).toHaveLength(3)
  })

  it('adds and removes tags on the selection', async () => {
    const changes: unknown[] = []
    const tagged = PHOTOS.map((p) => ({ ...p, tags: ['beach'] }))
    stubApi({
      ...libraryRoutes(() => tagged),
      'POST /api/library/photos/tags': async (req) => {
        changes.push(await req.json())
        return Response.json([])
      },
    })
    render(<Harness />)
    await userEvent.click((await thumbs())[0])
    const info = screen.getByRole('complementary', { name: 'Photo info' })
    await userEvent.type(
      within(info).getByLabelText('Add a tag'),
      'Lisbon{Enter}',
    )
    await userEvent.click(
      within(info).getByRole('button', { name: 'Remove tag beach' }),
    )
    await waitFor(() => expect(changes).toHaveLength(2))
    expect(changes).toEqual([
      { ids: ['aaaaaaaaaaaa'], add: ['Lisbon'], remove: [] },
      { ids: ['aaaaaaaaaaaa'], add: [], remove: ['beach'] },
    ])
  })
})
