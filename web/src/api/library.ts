import { api } from './client'
import { ApiError, detail, isSupportedFile } from './documents'
import type { components, operations } from './schema'

export type LibraryPhoto = components['schemas']['LibraryPhoto']
export type ImportResult = components['schemas']['ImportResult']
export type Skipped = components['schemas']['Skipped']
export type IndexStatus = components['schemas']['IndexStatus']
export type Album = components['schemas']['Album']
export type Marks = components['schemas']['Marks']
export type Facets = components['schemas']['Facets']
export type Facet = components['schemas']['Facet']
export type SearchResult = components['schemas']['SearchResult']
export type TagCount = components['schemas']['TagCount']
export type Flag = LibraryPhoto['flag']
export type Label = NonNullable<LibraryPhoto['label']>
export type PhotoFilter = NonNullable<
  operations['listLibraryPhotos']['parameters']['query']
>

export const LABELS: Label[] = ['red', 'yellow', 'green', 'blue', 'purple']

/** Files per import request, and roughly how many bytes, so big imports go in pieces. */
export const IMPORT_BATCH_FILES = 12
export const IMPORT_BATCH_BYTES = 48 * 1024 * 1024

/** Split files into import requests of a sensible size. */
export function importBatches(files: File[]): File[][] {
  const batches: File[][] = []
  let batch: File[] = []
  let bytes = 0
  for (const file of files) {
    if (
      batch.length > 0 &&
      (batch.length >= IMPORT_BATCH_FILES ||
        bytes + file.size > IMPORT_BATCH_BYTES)
    ) {
      batches.push(batch)
      batch = []
      bytes = 0
    }
    batch.push(file)
    bytes += file.size
  }
  if (batch.length) batches.push(batch)
  return batches
}

/** Photos among dropped or picked files (a folder holds other things too). */
export function photoFiles(files: Iterable<File>): File[] {
  return [...files].filter(
    (f) => isSupportedFile(f) && !/\.photoagent$/i.test(f.name),
  )
}

export async function importPhotos(files: File[]): Promise<ImportResult> {
  const form = new FormData()
  for (const file of files) form.append('files', file, file.name)
  const { data, error, response } = await api.POST('/api/library/photos', {
    body: form as unknown as { files: string[] },
    bodySerializer: (body) => body as unknown as FormData,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** A filter without its empty conditions, as query parameters. */
export function filterQuery(filter: PhotoFilter): PhotoFilter {
  return Object.fromEntries(
    Object.entries(filter).filter(
      ([, v]) =>
        v !== null &&
        v !== undefined &&
        v !== '' &&
        v !== 0 &&
        !(Array.isArray(v) && v.length === 0),
    ),
  ) as PhotoFilter
}

export async function listPhotos(
  filter: PhotoFilter = {},
): Promise<LibraryPhoto[]> {
  const { data, error, response } = await api.GET('/api/library/photos', {
    params: { query: filterQuery(filter) },
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Rate, flag, or label photos; returns them as they are now. */
export async function markPhotos(marks: Marks): Promise<LibraryPhoto[]> {
  const { data, error, response } = await api.POST(
    '/api/library/photos/marks',
    { body: marks },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data.photos
}

export async function fetchFacets(): Promise<Facets> {
  const { data, error, response } = await api.GET('/api/library/facets')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function searchLibrary(
  query: string,
  album: string | null = null,
): Promise<SearchResult> {
  const { data, error, response } = await api.POST('/api/library/search', {
    body: { query, album },
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function changeTags(
  ids: string[],
  change: { add?: string[]; remove?: string[] },
): Promise<LibraryPhoto[]> {
  const { data, error, response } = await api.POST('/api/library/photos/tags', {
    body: { ids, add: change.add ?? [], remove: change.remove ?? [] },
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function listAlbums(): Promise<Album[]> {
  const { data, error, response } = await api.GET('/api/library/albums')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function createAlbum(
  name: string,
  photoIds: string[] = [],
): Promise<Album> {
  const { data, error, response } = await api.POST('/api/library/albums', {
    body: { name, photo_ids: photoIds },
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function renameAlbum(id: string, name: string): Promise<Album> {
  const { data, error, response } = await api.PATCH(
    '/api/library/albums/{album_id}',
    { params: { path: { album_id: id } }, body: { name } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function deleteAlbum(id: string): Promise<void> {
  const { error, response } = await api.DELETE(
    '/api/library/albums/{album_id}',
    { params: { path: { album_id: id } } },
  )
  if (!response.ok) throw new ApiError(detail(error, response.status))
}

export async function addToAlbum(id: string, ids: string[]): Promise<Album> {
  const { data, error, response } = await api.POST(
    '/api/library/albums/{album_id}/photos',
    { params: { path: { album_id: id } }, body: { ids } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function removeFromAlbum(
  id: string,
  ids: string[],
): Promise<Album> {
  const { data, error, response } = await api.POST(
    '/api/library/albums/{album_id}/photos/remove',
    { params: { path: { album_id: id } }, body: { ids } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function removePhotos(ids: string[]): Promise<number> {
  const { data, error, response } = await api.POST(
    '/api/library/photos/remove',
    { body: { ids } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data.removed
}

export async function libraryStatus(): Promise<IndexStatus> {
  const { data, error, response } = await api.GET('/api/library/status')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** URL of a small JPEG of the photo as it looks now; changes when it is edited. */
export function thumbnailUrl(photo: LibraryPhoto): string {
  return `/api/library/photos/${photo.id}/thumbnail?revision=${photo.revision}`
}

/** URL of the photo with its edits at preview size, for the loupe. */
export function largeUrl(photo: LibraryPhoto): string {
  return `/api/documents/${photo.id}/preview?revision=${photo.revision}`
}

export type Person = components['schemas']['Person']
export type Face = components['schemas']['Face']
export type LibrarySettings = components['schemas']['LibrarySettings']

export async function librarySettings(): Promise<LibrarySettings> {
  const { data, error, response } = await api.GET('/api/library/settings')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function changeLibrarySettings(
  settings: LibrarySettings,
): Promise<LibrarySettings> {
  const { data, error, response } = await api.PUT('/api/library/settings', {
    body: settings,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function listPeople(): Promise<Person[]> {
  const { data, error, response } = await api.GET('/api/library/people')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function renamePerson(
  id: string,
  name: string | null,
): Promise<Person> {
  const { data, error, response } = await api.PATCH(
    '/api/library/people/{person_id}',
    { params: { path: { person_id: id } }, body: { name } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function mergePeople(id: string, into: string): Promise<Person> {
  const { data, error, response } = await api.POST(
    '/api/library/people/{person_id}/merge',
    { params: { path: { person_id: id } }, body: { into } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function forgetPerson(id: string): Promise<void> {
  const { error, response } = await api.DELETE(
    '/api/library/people/{person_id}',
    { params: { path: { person_id: id } } },
  )
  if (!response.ok) throw new ApiError(detail(error, response.status))
}

export async function listFaces(photoId: string): Promise<Face[]> {
  const { data, error, response } = await api.GET(
    '/api/library/photos/{photo_id}/faces',
    { params: { path: { photo_id: photoId } } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Say who a face is, or (null) that it is not the person it was grouped with. */
export async function changeFace(
  faceId: string,
  person: string | null,
): Promise<Face> {
  const { data, error, response } = await api.POST(
    '/api/library/faces/{face_id}',
    { params: { path: { face_id: faceId } }, body: { person } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export function faceUrl(faceId: string): string {
  return `/api/library/faces/${faceId}/thumbnail`
}

export type DuplicateGroup = components['schemas']['DuplicateGroup']

export async function listDuplicates(): Promise<DuplicateGroup[]> {
  const { data, error, response } = await api.GET('/api/library/duplicates')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** These photos are different, however alike: stop grouping them. */
export async function dismissDuplicates(ids: string[]): Promise<void> {
  const { error, response } = await api.POST(
    '/api/library/duplicates/dismiss',
    {
      body: { ids },
    },
  )
  if (!response.ok) throw new ApiError(detail(error, response.status))
}

export type CullRequest = components['schemas']['CullRequest']
export type CullResult = components['schemas']['CullResult']

/** Pick the best photos: one per moment, with the reasons for each. */
export async function cullPhotos(request: CullRequest): Promise<CullResult> {
  const { data, error, response } = await api.POST('/api/library/cull', {
    body: request,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}
