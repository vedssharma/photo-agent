import { useEffect, useState } from 'react'

import {
  type DocumentView,
  type Gamut,
  type Proof,
  type ProofSpace,
  fetchGamut,
} from '../api/documents'

interface Props {
  doc: DocumentView
  proof: Proof | null
  onProof: (proof: Proof | null) => void
  /** Overridable for tests. */
  loadGamut?: (docId: string) => Promise<Gamut>
}

const SPACES: Record<ProofSpace, string> = {
  srgb: 'sRGB',
  'display-p3': 'Display P3',
}

function share(fraction: number): string {
  if (fraction <= 0) return 'none'
  if (fraction < 0.01) return 'under 1%'
  return `${Math.round(fraction * 100)}%`
}

/**
 * Soft-proofing for wide-gamut photos (RAW and Display P3 ones): preview the photo as it
 * will look saved in sRGB or Display P3, and optionally paint the colors that will not fit.
 */
export function ProofControl({
  doc,
  proof,
  onProof,
  loadGamut = fetchGamut,
}: Props) {
  const [gamut, setGamut] = useState<Gamut | null>(null)
  useEffect(() => {
    let current = true
    loadGamut(doc.id)
      .then((found) => current && setGamut(found))
      .catch(() => current && setGamut(null))
    return () => {
      current = false
    }
  }, [doc.id, doc.revision, loadGamut])

  const outside =
    proof && gamut
      ? proof.space === 'srgb'
        ? gamut.outside_srgb
        : gamut.outside_display_p3
      : null
  return (
    <span className="proof">
      <label>
        Proof{' '}
        <select
          aria-label="Soft proof"
          value={proof?.space ?? ''}
          onChange={(e) =>
            onProof(
              e.target.value
                ? {
                    space: e.target.value as ProofSpace,
                    warn: proof?.warn ?? true,
                  }
                : null,
            )
          }
        >
          <option value="">Off (full color)</option>
          {Object.entries(SPACES).map(([value, name]) => (
            <option key={value} value={value}>
              {name}
            </option>
          ))}
        </select>
      </label>
      {proof && (
        <label title="Paint the colors that do not fit in magenta">
          <input
            type="checkbox"
            checked={proof.warn}
            onChange={(e) => onProof({ ...proof, warn: e.target.checked })}
          />
          Show colors that won’t fit
        </label>
      )}
      {proof && outside !== null && (
        <span className="status">
          {share(outside)} outside {SPACES[proof.space]}
        </span>
      )}
    </span>
  )
}
