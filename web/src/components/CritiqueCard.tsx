import type { Critique } from '../api/advice'

const ASPECT_LABEL: Record<Critique['points'][number]['aspect'], string> = {
  composition: 'Composition',
  exposure: 'Light',
  color: 'Color',
  sharpness: 'Sharpness',
  subject: 'Subject',
  mood: 'Mood',
}

interface Props {
  critique: Critique
  disabled?: boolean
  /** Ask the agent to carry out a point's fix. */
  onFix: (request: string) => void
}

/** What works and what does not, with a one-click fix for each point to improve. */
export function CritiqueCard({ critique, disabled = false, onFix }: Props) {
  return (
    <ul className="critique" aria-label="Feedback">
      {critique.points.map((point, i) => (
        <li key={i} className={point.verdict}>
          <span className="critique-aspect">
            {point.verdict === 'good' ? '✓' : '•'} {ASPECT_LABEL[point.aspect]}
          </span>{' '}
          {point.text}
          {point.fix && (
            <button
              type="button"
              disabled={disabled}
              title={point.fix}
              onClick={() => onFix(point.fix!)}
            >
              {point.fix_label ?? 'Fix it'}
            </button>
          )}
        </li>
      ))}
    </ul>
  )
}
