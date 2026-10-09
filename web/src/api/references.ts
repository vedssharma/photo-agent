import { api } from './client'
import { ApiError, type DocumentView, detail } from './documents'
import type { components } from './schema'

export type Reference = components['schemas']['Reference']

/** Share another photo to match this one to. */
export async function uploadReference(
  docId: string,
  file: File,
): Promise<Reference> {
  const form = new FormData()
  form.append('file', file, file.name)
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/references',
    {
      params: { path: { doc_id: docId } },
      body: form as unknown as { file: string },
      bodySerializer: (body) => body as unknown as FormData,
    },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export function referenceUrl(docId: string, refId: string): string {
  return `/api/documents/${docId}/references/${refId}`
}

/** Add a layer matching the photo's color and tone to a shared reference. */
export async function matchReference(
  docId: string,
  refId: string,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/references/{ref_id}/match',
    { params: { path: { doc_id: docId, ref_id: refId } } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}
