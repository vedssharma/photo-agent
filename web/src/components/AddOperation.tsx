import type { OperationSpec } from '../api/operations'

interface Props {
  specs: OperationSpec[]
  label: string
  disabled?: boolean
  onAdd: (spec: OperationSpec) => void
}

const GROUP_NAMES: Record<OperationSpec['group'], string> = {
  light: 'Light',
  color: 'Color',
  detail: 'Detail',
  finishing: 'Finishing',
  framing: 'Crop & rotate',
  retouch: 'Retouch',
}

/** A picker that adds a new operation with neutral settings. */
export function AddOperation({ specs, label, disabled = false, onAdd }: Props) {
  const groups = [...new Set(specs.map((s) => s.group))]
  return (
    <select
      className="add-op"
      aria-label={label}
      value=""
      disabled={disabled || specs.length === 0}
      onChange={(e) => {
        const spec = specs.find((s) => s.op === e.target.value)
        if (spec) onAdd(spec)
      }}
    >
      <option value="">{label}…</option>
      {groups.map((group) => (
        <optgroup key={group} label={GROUP_NAMES[group]}>
          {specs
            .filter((s) => s.group === group)
            .map((s) => (
              <option key={s.op} value={s.op}>
                {s.label}
              </option>
            ))}
        </optgroup>
      ))}
    </select>
  )
}
