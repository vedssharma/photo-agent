import type { EditState, Layer } from '../api/documents'

/** Helpers that return an updated copy of an edit state, for manual edits. */

export function newLayerId(): string {
  const hex = Array.from(crypto.getRandomValues(new Uint8Array(4)), (b) =>
    b.toString(16).padStart(2, '0'),
  ).join('')
  return `L${hex.slice(0, 7)}`
}

export function updateLayer(
  state: EditState,
  layerId: string,
  changes: Partial<Layer>,
): EditState {
  return {
    ...state,
    layers: state.layers.map((layer) =>
      layer.id === layerId ? { ...layer, ...changes } : layer,
    ),
  }
}

export function removeLayer(state: EditState, layerId: string): EditState {
  return {
    ...state,
    layers: state.layers.filter((layer) => layer.id !== layerId),
  }
}

/** Move a layer up (toward the top of the stack, +1) or down (-1). */
export function moveLayer(
  state: EditState,
  layerId: string,
  direction: 1 | -1,
): EditState {
  const layers = [...state.layers]
  const from = layers.findIndex((layer) => layer.id === layerId)
  const to = from + direction
  if (from < 0 || to < 0 || to >= layers.length) return state
  ;[layers[from], layers[to]] = [layers[to], layers[from]]
  return { ...state, layers }
}
