import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { type Proof, previewUrl } from '../api/documents'
import { makeDoc } from '../test/fixtures'
import { ProofControl } from './ProofControl'

const GAMUT = {
  color_space: 'rec2020' as const,
  outside_srgb: 0.12,
  outside_display_p3: 0.004,
}

describe('ProofControl', () => {
  it('turns soft-proofing on with the gamut warning and says how much is outside', async () => {
    const onProof = vi.fn()
    const doc = makeDoc({ color_space: 'rec2020' })
    const loadGamut = vi.fn().mockResolvedValue(GAMUT)
    const { rerender } = render(
      <ProofControl
        doc={doc}
        proof={null}
        onProof={onProof}
        loadGamut={loadGamut}
      />,
    )
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Soft proof' }),
      'srgb',
    )
    expect(onProof).toHaveBeenCalledWith({ space: 'srgb', warn: true })

    const proof: Proof = { space: 'srgb', warn: true }
    rerender(
      <ProofControl
        doc={doc}
        proof={proof}
        onProof={onProof}
        loadGamut={loadGamut}
      />,
    )
    expect(await screen.findByText('12% outside sRGB')).toBeInTheDocument()
    await userEvent.click(
      screen.getByRole('checkbox', { name: /colors that won’t fit/ }),
    )
    expect(onProof).toHaveBeenLastCalledWith({ space: 'srgb', warn: false })

    rerender(
      <ProofControl
        doc={doc}
        proof={{ space: 'display-p3', warn: true }}
        onProof={onProof}
        loadGamut={loadGamut}
      />,
    )
    expect(
      await screen.findByText('under 1% outside Display P3'),
    ).toBeInTheDocument()
  })

  it('asks for the proofed preview', () => {
    const doc = makeDoc({ revision: 'r1' })
    expect(previewUrl(doc)).toBe(
      '/api/documents/abc123abc123/preview?revision=r1',
    )
    expect(previewUrl(doc, { space: 'display-p3', warn: true })).toBe(
      '/api/documents/abc123abc123/preview?revision=r1&proof=display-p3&gamut_warning=true',
    )
    expect(previewUrl(doc, { space: 'srgb', warn: false })).toBe(
      '/api/documents/abc123abc123/preview?revision=r1&proof=srgb',
    )
  })
})
