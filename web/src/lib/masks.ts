import type { Mask, MaskKind } from '../api/documents'

export type SemanticMask = Extract<Mask, { kind: 'semantic' }>
export type SemanticTarget = SemanticMask['target']

export const MASK_KINDS: { kind: MaskKind; label: string }[] = [
  { kind: 'brush', label: 'Brush' },
  { kind: 'linear', label: 'Linear gradient' },
  { kind: 'radial', label: 'Radial gradient' },
  { kind: 'luminosity', label: 'Brightness range' },
]

/** Things an AI model can find, grouped for the mask picker. */
export const SEMANTIC_TARGETS: {
  target: SemanticTarget
  label: string
  group: 'Find with AI' | 'Portrait'
}[] = [
  { target: 'subject', label: 'Main subject', group: 'Find with AI' },
  { target: 'people', label: 'People', group: 'Find with AI' },
  { target: 'sky', label: 'Sky', group: 'Find with AI' },
  { target: 'object', label: 'An object (click it)', group: 'Find with AI' },
  { target: 'skin', label: 'Skin', group: 'Portrait' },
  { target: 'face', label: 'Face', group: 'Portrait' },
  { target: 'eyes', label: 'Eyes', group: 'Portrait' },
  { target: 'lips', label: 'Lips', group: 'Portrait' },
  { target: 'teeth', label: 'Teeth', group: 'Portrait' },
  { target: 'hair', label: 'Hair', group: 'Portrait' },
]

/** The picker's value for a mask: its kind, or "semantic:<target>". */
export function maskChoice(mask: Mask | null): string {
  if (!mask) return ''
  return mask.kind === 'semantic' ? `semantic:${mask.target}` : mask.kind
}

export function maskLabel(kind: MaskKind, target?: SemanticTarget): string {
  if (kind === 'semantic')
    return SEMANTIC_TARGETS.find((t) => t.target === target)?.label ?? 'AI'
  return MASK_KINDS.find((m) => m.kind === kind)?.label ?? kind
}

export function semanticMask(target: SemanticTarget): SemanticMask {
  return {
    kind: 'semantic',
    target,
    points: [],
    strokes: [],
    description: '',
    invert: false,
  }
}

/** A sensible starting mask of each kind, to adjust from. */
export function defaultMask(kind: Exclude<MaskKind, 'semantic'>): Mask {
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
  /** Waiting for a click on the photo to select an object for the selected layer. */
  picking: boolean
  /** Painting touch-ups onto an AI selection instead of clicking to select. */
  refining: boolean
}

export const DEFAULT_MASK_TOOL: MaskTool = {
  show: false,
  size: 0.04,
  hardness: 50,
  erase: false,
  picking: false,
  refining: false,
}

/** Whether a mask is placed by drawing or clicking on the photo (with this tool). */
export function isDrawn(mask: Mask, tool?: MaskTool): boolean {
  if (mask.kind === 'semantic')
    return mask.target === 'object' || (tool?.refining ?? false)
  return mask.kind !== 'luminosity'
}

/** Whether pressing on the photo paints strokes into this mask. */
export function isPainted(mask: Mask, tool: MaskTool): boolean {
  return mask.kind === 'brush' || (mask.kind === 'semantic' && tool.refining)
}

/** An object selection with nothing picked yet, for the first click. */
export const EMPTY_OBJECT: SemanticMask = semanticMask('object')
