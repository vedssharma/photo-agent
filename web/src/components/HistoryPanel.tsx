import { useState } from 'react'

import { type DocumentView, type StepView, checkout } from '../api/documents'
import { historyRows } from '../lib/history'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  /** True while the agent is editing; history is locked until it finishes. */
  disabled?: boolean
}

const KIND_ICON: Record<StepView['kind'], string> = {
  agent: '✦',
  manual: '✎',
}

const KIND_NAME: Record<StepView['kind'], string> = {
  agent: 'Agent edit',
  manual: 'Manual edit',
}

/** Every step as a named entry; click one to jump there. Branches are indented. */
export function HistoryPanel({ doc, onDocument, disabled = false }: Props) {
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function jump(stepId: string | null) {
    if (stepId === doc.head) return
    setWorking(true)
    setError(null)
    try {
      onDocument(await checkout(doc.id, stepId))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  const rows = historyRows(doc.history)
  const locked = disabled || working
  return (
    <section className="panel history-panel" aria-label="History">
      <h2>History</h2>
      <ol className="history-list">
        <li>
          <button
            type="button"
            className="history-step"
            aria-current={doc.head === null ? 'step' : undefined}
            disabled={locked}
            onClick={() => jump(null)}
          >
            <span className="kind" aria-hidden="true">
              ◯
            </span>
            Original
          </button>
        </li>
        {rows.map(({ step, depth }) => (
          <li key={step.id} style={{ paddingLeft: `${depth}rem` }}>
            <button
              type="button"
              className={`history-step${step.active ? '' : ' inactive'}`}
              aria-current={doc.head === step.id ? 'step' : undefined}
              title={`${KIND_NAME[step.kind]}: ${step.label}`}
              disabled={locked}
              onClick={() => jump(step.id)}
            >
              <span className="kind" aria-label={KIND_NAME[step.kind]}>
                {KIND_ICON[step.kind]}
              </span>
              {step.label}
            </button>
          </li>
        ))}
      </ol>
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
