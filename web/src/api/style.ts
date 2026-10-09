import { api } from './client'
import { ApiError, type DocumentView, detail } from './documents'
import type { components } from './schema'

export type StyleSummary = components['schemas']['StyleSummary']

/** What the app has learned about the person's taste. */
export async function fetchStyle(): Promise<StyleSummary> {
  const { data, error, response } = await api.GET('/api/style')
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export async function forgetStyle(): Promise<void> {
  const { response } = await api.DELETE('/api/style')
  if (!response.ok)
    throw new ApiError(`The server answered ${response.status}.`)
}

/** Add "my usual look" as one step in the photo's history. */
export async function applyUsualLook(docId: string): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/style/usual-look',
    { params: { path: { doc_id: docId } } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}
