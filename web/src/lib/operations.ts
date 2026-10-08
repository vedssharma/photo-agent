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

/** A short description like "Exposure (stops +0.4)", matching the server's wording. */
export function opSummary(op: Operation): string {
  const params = Object.entries(op)
    .filter(([key]) => key !== 'id' && key !== 'op')
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
