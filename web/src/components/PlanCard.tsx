import type { components } from '../api/schema'

type Plan = components['schemas']['Plan']

const KIND_NOTE: Record<Plan['steps'][number]['kind'], string | null> = {
  adjust: null,
  ai: 'AI model',
  generative: 'Generates new pixels',
}

interface Props {
  plan: Plan
  /** Whether the plan still waits for the go-ahead. */
  pending: boolean
  disabled?: boolean
  onApprove: () => void
}

/** The steps the agent means to take, with a go-ahead button while it waits. */
export function PlanCard({
  plan,
  pending,
  disabled = false,
  onApprove,
}: Props) {
  return (
    <div className="plan" aria-label="Plan">
      <ol>
        {plan.steps.map((step, i) => (
          <li key={i}>
            {step.text}
            {KIND_NOTE[step.kind] && (
              <span className={`plan-kind ${step.kind}`}>
                {KIND_NOTE[step.kind]}
              </span>
            )}
          </li>
        ))}
      </ol>
      {pending && (
        <div className="plan-actions">
          <button
            type="button"
            className="primary"
            disabled={disabled}
            onClick={onApprove}
          >
            Go ahead
          </button>
          <span className="fine-print">Or tell me what to change.</span>
        </div>
      )}
    </div>
  )
}
