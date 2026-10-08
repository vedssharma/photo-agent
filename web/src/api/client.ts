import createClient from 'openapi-fetch'

import type { paths } from './schema'

/**
 * Typed client for the photo-agent backend, generated from api/openapi.json.
 * Requests go to the page's own origin; in development Vite proxies /api to
 * the FastAPI server. `fetch` is looked up per request so tests can stub it.
 */
export const api = createClient<paths>({
  baseUrl: window.location.origin,
  fetch: (request) => globalThis.fetch(request),
})
