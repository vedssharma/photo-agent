import type { StepView } from '../api/documents'

export interface HistoryRow {
  step: StepView
  /** How far the row is indented: each branch off the main line adds one. */
  depth: number
}

/**
 * Flatten the history tree into rows, oldest first. A step's first child continues its
 * line; later children (branches made after going back) are indented under it.
 */
export function historyRows(steps: StepView[]): HistoryRow[] {
  const children = new Map<string | null, StepView[]>()
  for (const step of steps) {
    const siblings = children.get(step.parent) ?? []
    siblings.push(step)
    children.set(step.parent, siblings)
  }
  const rows: HistoryRow[] = []
  const visit = (parent: string | null, depth: number) => {
    const kids = children.get(parent) ?? []
    // Later branches are listed first so the original line reads straight down below them.
    kids.forEach((step, i) => {
      if (i === 0) return
      rows.push({ step, depth: depth + 1 })
      visit(step.id, depth + 1)
    })
    if (kids.length > 0) {
      rows.push({ step: kids[0], depth })
      visit(kids[0].id, depth)
    }
  }
  visit(null, 0)
  return rows
}
