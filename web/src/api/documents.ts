import { api } from './client'
import type { components } from './schema'

export type DocumentView = components['schemas']['DocumentView']
export type Operation = DocumentView['operations'][number]

/** File types the backend can decode. HEIC often has no MIME type in browsers. */
export const ACCEPTED_TYPES =
  '.jpg,.jpeg,.png,.heic,.heif,image/jpeg,image/png,image/heic'

const ACCEPTED_EXTENSIONS = /\.(jpe?g|png|heic|heif)$/i

export function isSupportedFile(file: File): boolean {
  return (
    ['image/jpeg', 'image/png', 'image/heic', 'image/heif'].includes(
      file.type,
    ) || ACCEPTED_EXTENSIONS.test(file.name)
  )
}

export class ApiError extends Error {}

function detail(error: unknown, status: number): string {
  if (error && typeof error === 'object' && 'detail' in error) {
    const d = (error as { detail: unknown }).detail
    if (typeof d === 'string') return d
  }
  return `The server answered ${status}.`
}

export async function uploadDocument(file: File): Promise<DocumentView> {
  const form = new FormData()
  form.append('file', file, file.name)
  const { data, error, response } = await api.POST('/api/documents', {
    // openapi-fetch would JSON-encode the body; send the form as-is.
    body: form as unknown as { file: string },
    bodySerializer: (body) => body as unknown as FormData,
  })
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** URL of the unedited photo at preview size. */
export function originalUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/original`
}

/** URL of the edited photo at preview size; changes whenever the edits do. */
export function previewUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/preview?revision=${doc.revision}`
}
