import type { Operation } from '../api/documents'
import type { ParamSpec } from '../api/operations'

/** "white_balance" → "White balance" */
export function humanize(name: string): string {
  const words = name.replace(/_/g, ' ')
  return words.charAt(0).toUpperCase() + words.slice(1)
}

function format(value: unknown): string {
  if (typeof value === 'number') {
    const text = String(Number(value.toPrecision(6)))
    return value > 0 ? `+${text}` : text
  }
  return Array.isArray(value) ? JSON.stringify(value) : String(value)
}

/** Fields the app records on generative operations, not shown as parameters. */
const APP_FIELDS = new Set(['id', 'op', 'model'])

/** A short description like "Exposure (stops +0.4)", matching the server's wording. */
export function opSummary(op: Operation): string {
  if ('prompt' in op || 'seed' in op) {
    const prompt = 'prompt' in op && op.prompt ? ` “${String(op.prompt)}”` : ''
    const seed = 'seed' in op && op.seed != null ? ` (seed ${op.seed})` : ''
    return `${humanize(op.op)}${prompt}${seed}`
  }
  const params = Object.entries(op)
    .filter(([key]) => !APP_FIELDS.has(key))
    .map(([key, value]) => `${key} ${format(value)}`)
  const title = humanize(op.op)
  return params.length > 0 ? `${title} (${params.join(', ')})` : title
}

/** A slider value as shown next to it: signed when the range is, two decimals for fine steps. */
export function formatValue(param: ParamSpec, value: number): string {
  const decimals = (param.step ?? 1) < 1 ? 2 : 0
  const text = value.toFixed(decimals)
  return (param.min ?? 0) < 0 && value > 0 ? `+${text}` : text
}

/** A random seed for a generative edit, in the range the server accepts. */
export function newSeed(): number {
  return crypto.getRandomValues(new Uint32Array(1))[0] & 0x7fffffff
}

/** Whether an operation changes what is in the photo, so it stands alone in its layer. */
export function isContentOp(op: Operation): boolean {
  return CONTENT_OPS.has(op.op)
}

export const CONTENT_OPS = new Set<string>([
  'remove',
  'generate',
  'replace_background',
  'relight',
  'restore_faces',
  'colorize',
])
