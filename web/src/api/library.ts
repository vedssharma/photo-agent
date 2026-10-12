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
