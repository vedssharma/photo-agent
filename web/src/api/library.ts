import { api } from './client'
import { ApiError, detail, isSupportedFile } from './documents'
import type { components } from './schema'

export type LibraryPhoto = components['schemas']['LibraryPhoto']
export type ImportResult = components['schemas']['ImportResult']
export type Skipped = components['schemas']['Skipped']
export type IndexStatus = components['schemas']['IndexStatus']

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

export async function listPhotos(): Promise<LibraryPhoto[]> {
  const { data, error, response } = await api.GET('/api/library/photos')
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
