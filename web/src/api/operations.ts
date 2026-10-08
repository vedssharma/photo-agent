import { api } from './client'
import type { components } from './schema'

export type OperationSpec = components['schemas']['OperationSpec']
export type ParamSpec = OperationSpec['params'][number]

let cached: Promise<OperationSpec[]> | null = null

/** Every operation with its parameter ranges, fetched once per page load. */
export function fetchOperationSpecs(): Promise<OperationSpec[]> {
  cached ??= api.GET('/api/operations').then(({ data, response }) => {
    if (!data) {
      cached = null
      throw new Error(`The server answered ${response.status}.`)
    }
    return data
  })
  return cached
}
