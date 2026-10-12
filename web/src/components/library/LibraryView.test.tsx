import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { ExportedFile } from '../../api/documents'
import { useImporter } from '../../hooks/useImporter'
import { libraryRoutes, makePhoto, stubApi } from '../../test/fixtures'
import { LibraryView } from './LibraryView'

function Harness({
  onEdit = vi.fn(),
  save = vi.fn(),
}: {
  onEdit?: (id: string) => void
  save?: (file: ExportedFile) => void
}) {
  const importer = useImporter()
  return (
    <LibraryView importer={importer} onEdit={onEdit} pollMs={10} save={save} />
  )
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
  it('turns on people albums, names someone, and shows their photos', async () => {
    let on = false
    const maya = {
      id: 'person1',
      name: null as string | null,
      photos: 2,
      cover: 'face1',
    }
    const settings = vi.fn()
    const renamed = vi.fn()
    const fetchMock = stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/library/settings': () => Response.json({ faces: on }),
      'PUT /api/library/settings': async (req) => {
        settings(await req.json())
        on = true
        return Response.json({ faces: true })
      },
      'GET /api/library/people': () => Response.json(on ? [maya] : []),
      'PATCH /api/library/people/person1': async (req) => {
        const body = (await req.json()) as { name: string }
        renamed(body)
        maya.name = body.name
        return Response.json(maya)
      },
    })
    vi.stubGlobal(
      'prompt',
      vi.fn(() => 'Maya'),
    )
    render(<Harness />)
    await thumbs()
    const people = screen.getByRole('region', { name: 'People' })
    expect(
      within(people).getByText(/on this computer only/),
    ).toBeInTheDocument()
    await userEvent.click(
      within(people).getByRole('button', { name: 'Find people' }),
    )
    expect(settings).toHaveBeenCalledWith({ faces: true })

    await userEvent.click(
      await within(people).findByRole('button', { name: /Unnamed/ }),
    )
    await userEvent.click(within(people).getByRole('button', { name: 'Name' }))
    expect(renamed).toHaveBeenCalledWith({ name: 'Maya' })
    expect(await screen.findByText('With Maya')).toBeInTheDocument()
    const urls = fetchMock.mock.calls.map(([r]) => (r as Request).url)
    expect(
      urls.some((u) => u.endsWith('/api/library/photos?people=person1')),
    ).toBe(true)
  })

  it('says who is in a photo and takes corrections', async () => {
    const changed = vi.fn()
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/library/settings': () => Response.json({ faces: true }),
      'GET /api/library/people': () =>
        Response.json([
          { id: 'person1', name: 'Maya', photos: 2, cover: 'face1' },
          { id: 'person2', name: null, photos: 3, cover: 'face9' },
        ]),
      'GET /api/library/photos/aaaaaaaaaaaa/faces': () =>
        Response.json([
          {
            id: 'face1',
            box: [0.1, 0.1, 0.3, 0.4],
            person: 'person1',
            person_name: 'Maya',
          },
        ]),
      'POST /api/library/faces/face1': async (req) => {
        changed(await req.json())
        return Response.json({
          id: 'face1',
          box: [0.1, 0.1, 0.3, 0.4],
          person: null,
        })
      },
    })
    render(<Harness />)
    await userEvent.click((await thumbs())[0])
    const info = screen.getByRole('complementary', { name: 'Photo info' })
    await userEvent.click(
      await within(info).findByRole('button', { name: 'This is not Maya' }),
    )
    expect(changed).toHaveBeenCalledWith({ person: null })
    await userEvent.selectOptions(
      within(info).getByLabelText('Who is this? (Maya)'),
      'Unnamed',
    )
    expect(changed).toHaveBeenLastCalledWith({ person: 'person2' })
  })
  it('reviews duplicates and keeps the chosen photo of each group', async () => {
    let groups = [
      {
        kind: 'near',
        keep: 'aaaaaaaaaaaa',
        photos: [PHOTOS[0], PHOTOS[1]],
      },
    ]
    const removed = vi.fn()
    const dismissed = vi.fn()
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/library/duplicates': () => Response.json(groups),
      'POST /api/library/photos/remove': async (req) => {
        removed(await req.json())
        groups = []
        return Response.json({ removed: 1 })
      },
      'POST /api/library/duplicates/dismiss': async (req) => {
        dismissed(await req.json())
        return new Response(null, { status: 204 })
      },
    })
    vi.stubGlobal(
      'confirm',
      vi.fn(() => true),
    )
    render(<Harness />)
    await thumbs()
    await userEvent.click(screen.getByRole('button', { name: 'Duplicates' }))
    const group = await screen.findByRole('region', { name: /Same shot/ })
    expect(
      within(group).getByRole('button', { name: 'Keep aaaaaaaaaaaa.jpg' }),
    ).toHaveAttribute('aria-pressed', 'true')

    await userEvent.click(
      within(group).getByRole('button', { name: 'Not duplicates' }),
    )
    expect(dismissed).toHaveBeenCalledWith({
      ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb'],
    })

    // Keep the other one instead, and delete the rest.
    await userEvent.click(
      within(group).getByRole('button', { name: 'Keep bbbbbbbbbbbb.jpg' }),
    )
    await userEvent.click(
      within(group).getByRole('button', { name: 'Delete the other 1 photo' }),
    )
    expect(removed).toHaveBeenCalledWith({ ids: ['aaaaaaaaaaaa'] })
    expect(await screen.findByText('No duplicates')).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'All photos' }))
    expect(await thumbs()).toHaveLength(3)
  })
  it('picks the best shots, says why, and flags them', async () => {
    const asked = vi.fn()
    const marked = vi.fn()
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'POST /api/library/cull': async (req) => {
        asked(await req.json())
        return Response.json({
          picks: [
            {
              photo: PHOTOS[1],
              score: 0.9,
              moment: 1,
              moment_size: 3,
              reasons: ['Best of 3 similar shots', 'sharpest of them'],
            },
          ],
          considered: 3,
          moments: 1,
          summary: 'Picked 1 of 3 photos from 1 moment: the best shot of each.',
          source: 'built-in',
        })
      },
      'POST /api/library/photos/marks': async (req) => {
        marked(await req.json())
        return Response.json({ changed: 1, photos: [] })
      },
    })
    vi.stubGlobal(
      'prompt',
      vi.fn(() => '1'),
    )
    render(<Harness />)
    await thumbs()
    await userEvent.click(
      screen.getByRole('button', { name: 'Pick the best…' }),
    )
    expect(asked).toHaveBeenCalledWith({
      count: 1,
      ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb', 'cccccccccccc'],
      explain: true,
    })
    const list = await screen.findByRole('list', { name: 'Best shots' })
    expect(
      within(list).getByText('Best of 3 similar shots, sharpest of them'),
    ).toBeInTheDocument()
    expect(screen.getByText(/Picked 1 of 3 photos/)).toBeInTheDocument()

    await userEvent.click(screen.getByRole('button', { name: 'Flag as picks' }))
    expect(marked).toHaveBeenCalledWith({
      ids: ['bbbbbbbbbbbb'],
      flag: 'picked',
    })
    await userEvent.click(
      screen.getByRole('button', { name: 'Back to photos' }),
    )
    expect(await thumbs()).toHaveLength(3)
  })
  it('edits the selection together from one instruction', async () => {
    const started = vi.fn()
    let polls = 0
    const status = {
      id: 'batch1',
      total: 2,
      reference: 'aaaaaaaaaaaa',
      failed: [],
      current: null,
      message: null as string | null,
    }
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'GET /api/recipes': () => Response.json([]),
      'POST /api/library/batch': async (req) => {
        started(await req.json())
        return Response.json(
          { ...status, state: 'running', done: 0 },
          { status: 202 },
        )
      },
      'GET /api/library/batch/batch1': () => {
        polls += 1
        return Response.json(
          polls < 2
            ? { ...status, state: 'running', done: 1, current: 'Editing b' }
            : {
                ...status,
                state: 'done',
                done: 2,
                message: 'Warm and bright.',
              },
        )
      },
    })
    render(<Harness />)
    const options = await thumbs()
    await userEvent.click(options[0])
    fireEvent.click(options[1], { shiftKey: true })
    await userEvent.click(
      screen.getByRole('button', { name: 'Edit together…' }),
    )

    const dialog = screen.getByRole('dialog', {
      name: 'Edit 2 photos together',
    })
    await userEvent.type(
      within(dialog).getByLabelText('Describe the look'),
      'make these consistent',
    )
    await userEvent.click(
      within(dialog).getByRole('button', { name: 'Edit 2 photos' }),
    )
    expect(started).toHaveBeenCalledWith({
      ids: ['aaaaaaaaaaaa', 'bbbbbbbbbbbb'],
      reference: 'aaaaaaaaaaaa',
      match: 60,
      instruction: 'make these consistent',
      recipe: null,
    })
    expect(
      await screen.findByText(
        /Edited 2 of 2 photos together. Warm and bright./,
      ),
    ).toBeInTheDocument()
  })

  it('downloads the selection as one file', async () => {
    const save = vi.fn()
    const asked = vi.fn()
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'POST /api/library/export': async (req) => {
        asked(await req.json())
        return new Response('zip', {
          headers: {
            'content-type': 'application/zip',
            'content-disposition': 'attachment; filename="photos-2.zip"',
          },
        })
      },
    })
    render(<Harness save={save} />)
    const options = await thumbs()
    await userEvent.click(options[1])
    fireEvent.click(options[2], { ctrlKey: true })
    await userEvent.click(screen.getByRole('button', { name: 'Download' }))
    await waitFor(() => expect(save).toHaveBeenCalled())
    expect(asked).toHaveBeenCalledWith({
      ids: ['bbbbbbbbbbbb', 'cccccccccccc'],
      options: {
        format: 'jpeg',
        quality: 92,
        keep_location: false,
        upscale: 1,
        color_space: 'srgb',
      },
    })
    expect(save.mock.calls[0][0].filename).toBe('photos-2.zip')
  })

  it('stitches a panorama from the selection, and says why when it cannot', async () => {
    const asked = vi.fn()
    let attempt = 0
    stubApi({
      ...libraryRoutes(() => PHOTOS),
      'POST /api/library/merge': async (req) => {
        asked(await req.json())
        attempt += 1
        return attempt === 1
          ? Response.json(makePhoto('dddddddddddd'), { status: 201 })
          : Response.json(
              { detail: 'These photos could not be stitched.' },
              { status: 422 },
            )
      },
    })
    render(<Harness />)
    const options = await thumbs()
    await userEvent.click(options[1])
    fireEvent.click(options[2], { ctrlKey: true })
    await userEvent.click(
      screen.getByRole('button', { name: 'Stitch panorama' }),
    )
    await waitFor(() =>
      expect(asked).toHaveBeenCalledWith({
        ids: ['bbbbbbbbbbbb', 'cccccccccccc'],
        kind: 'panorama',
      }),
    )
    await userEvent.click(options[1])
    fireEvent.click(options[2], { ctrlKey: true })
    await userEvent.click(screen.getByRole('button', { name: 'Merge to HDR' }))
    expect(
      await screen.findByText(
        /Could not merge to HDR: These photos could not be stitched./,
      ),
    ).toBeInTheDocument()
  })
})
