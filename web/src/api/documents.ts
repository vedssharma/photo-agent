import { api } from './client'
import type { components } from './schema'

export type DocumentView = components['schemas']['DocumentView']
export type Operation = DocumentView['operations'][number]
export type StepView = DocumentView['history'][number]

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

/** URL of the unedited look with the current crop and rotation, to compare against. */
export function beforeUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/before?revision=${doc.revision}`
}

/** URL of the edited photo at preview size; changes whenever the edits do. */
export function previewUrl(doc: DocumentView): string {
  return `/api/documents/${doc.id}/preview?revision=${doc.revision}`
}

export async function undo(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/undo',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function redo(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/redo',
    {
      params: { path: { doc_id: docId } },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

/** Show any step in the history, or the original photo when `stepId` is null. */
export async function checkout(
  docId: string,
  stepId: string | null,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/checkout',
    {
      params: { path: { doc_id: docId } },
      body: { step_id: stepId },
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export type ExportOptions = components['schemas']['ExportOptions']

export interface ExportedFile {
  blob: Blob
  filename: string
}

/** Parse the download name from a Content-Disposition header. */
export function dispositionFilename(
  header: string | null,
  fallback: string,
): string {
  const encoded = header?.match(/filename\*=UTF-8''([^;]+)/i)?.[1]
  if (encoded) return decodeURIComponent(encoded)
  return header?.match(/filename="?([^";]+)"?/i)?.[1] ?? fallback
}

export async function exportDocument(
  doc: DocumentView,
  options: ExportOptions,
): Promise<ExportedFile> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/export',
    {
      params: { path: { doc_id: doc.id } },
      body: options,
      parseAs: 'blob',
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  const ext = options.format === 'png' ? 'png' : 'jpg'
  return {
    blob: data as Blob,
    filename: dispositionFilename(
      response.headers.get('content-disposition'),
      `photo.${ext}`,
    ),
  }
}

/** Hand a file to the browser's download manager. */
export function saveFile({ blob, filename }: ExportedFile) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.append(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 10_000)
}
