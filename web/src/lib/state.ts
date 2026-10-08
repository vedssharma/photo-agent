import type { EditState, Layer } from '../api/documents'

/** Helpers that return an updated copy of an edit state, for manual edits. */

/** Selection value for the framing section (layer ids always start with "L"). */
export const FRAMING = 'framing'

/** A slider being dragged, for live previews before the change is committed. */
export interface Preview {
  /** The layer the dragged operation is in, or null for the framing. */
  layerId: string | null
  opId: string
  param: string
  value: number
}

function randomHex(bytes: number): string {
  return Array.from(crypto.getRandomValues(new Uint8Array(bytes)), (b) =>
    b.toString(16).padStart(2, '0'),
  ).join('')
}

/** Ids in the same shape the server makes. */
export function newLayerId(): string {
  return `L${randomHex(4).slice(0, 7)}`
}

export function newOpId(): string {
  return randomHex(4)
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

type AnyOperation = EditState['framing'][number] | Layer['operations'][number]

/** Change one operation (in the framing or any layer) by id. */
export function updateOperation(
  state: EditState,
  opId: string,
  changes: Record<string, unknown>,
): EditState {
  const update = <T extends AnyOperation>(op: T): T =>
    op.id === opId ? ({ ...op, ...changes } as T) : op
  return {
    framing: state.framing.map(update),
    layers: state.layers.map((layer) => ({
      ...layer,
      operations: layer.operations.map(update),
    })),
  }
}

export function removeOperation(state: EditState, opId: string): EditState {
  return {
    framing: state.framing.filter((op) => op.id !== opId),
    layers: state.layers.map((layer) => ({
      ...layer,
      operations: layer.operations.filter((op) => op.id !== opId),
    })),
  }
}

/** Add an operation to a layer, or to the framing when `layerId` is null. */
export function addOperation(
  state: EditState,
  layerId: string | null,
  op: AnyOperation,
): EditState {
  if (layerId === null)
    return {
      ...state,
      framing: [...state.framing, op as EditState['framing'][number]],
    }
  return updateLayer(state, layerId, {
    operations: [
      ...(state.layers.find((l) => l.id === layerId)?.operations ?? []),
      op as Layer['operations'][number],
    ],
  })
}

/** Add an empty layer on top of the stack. */
export function addLayer(state: EditState, layer: Layer): EditState {
  return { ...state, layers: [...state.layers, layer] }
}
