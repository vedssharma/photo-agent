import type { Mask, MaskKind } from '../api/documents'

export const MASK_KINDS: { kind: MaskKind; label: string }[] = [
  { kind: 'brush', label: 'Brush' },
  { kind: 'linear', label: 'Linear gradient' },
  { kind: 'radial', label: 'Radial gradient' },
  { kind: 'luminosity', label: 'Brightness range' },
]

export function maskLabel(kind: MaskKind): string {
  return MASK_KINDS.find((m) => m.kind === kind)?.label ?? kind
}

/** A sensible starting mask of each kind, to adjust from. */
export function defaultMask(kind: MaskKind): Mask {
  switch (kind) {
    case 'brush':
      return { kind, strokes: [], invert: false }
    case 'linear':
      // Top of the frame fading out by the middle: the usual sky gradient.
      return { kind, start: [0.5, 0], end: [0.5, 0.5], invert: false }
    case 'radial':
      return {
        kind,
        center: [0.5, 0.5],
        radius_x: 0.3,
        radius_y: 0.3,
        feather: 50,
        invert: false,
      }
    case 'luminosity':
      return { kind, low: 0.6, high: 1, feather: 0.1, invert: false }
  }
}

/** Settings for painting brush masks, shared by the panel and the canvas. */
export interface MaskTool {
  /** Tint the photo where the selected layer applies. */
  show: boolean
  /** Brush radius as a fraction of the photo's long edge. */
  size: number
  hardness: number
  erase: boolean
}

export const DEFAULT_MASK_TOOL: MaskTool = {
  show: false,
  size: 0.04,
  hardness: 50,
  erase: false,
}

/** Whether a mask kind is placed by drawing on the photo. */
export function isDrawn(kind: MaskKind): boolean {
  return kind !== 'luminosity'
}
