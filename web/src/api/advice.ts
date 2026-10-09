import { api } from './client'
import { ApiError, type DocumentView, detail } from './documents'
import type { components } from './schema'

export type SuggestionSet = components['schemas']['SuggestionSet']
export type Suggestion = components['schemas']['Suggestion']

/** A few directions the photo could go in, to pick from. */
export async function fetchSuggestions(
  docId: string,
  refresh = false,
): Promise<SuggestionSet> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/suggestions',
    { params: { path: { doc_id: docId } }, body: { refresh } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}

export function suggestionPreviewUrl(
  doc: DocumentView,
  suggestion: Suggestion,
): string {
  return `/api/documents/${doc.id}/suggestions/${suggestion.id}/preview?revision=${doc.revision}`
}

/** Add a suggestion's layers, as one step the agent knows about. */
export async function applySuggestion(
  docId: string,
  suggestionId: string,
): Promise<DocumentView> {
  const { data, error, response } = await api.POST(
    '/api/documents/{doc_id}/suggestions/{suggestion_id}',
    { params: { path: { doc_id: docId, suggestion_id: suggestionId } } },
  )
  if (!data) throw new ApiError(detail(error, response.status))
  return data
}
