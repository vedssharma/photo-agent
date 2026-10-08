import createClient from 'openapi-fetch'

import type { paths } from './schema'

/** Typed client for the photo-agent backend, generated from api/openapi.json. */
export const api = createClient<paths>({ baseUrl: '/' })
