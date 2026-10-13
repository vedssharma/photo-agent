import { render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'

import { LensNote } from './LensNote'

describe('LensNote', () => {
  it('names the profile in use', async () => {
    const load = vi.fn().mockResolvedValue({
      camera: 'Canon Canon EOS 5D Mark III',
      lens: 'EF 24-105mm f/4L IS USM',
      focal_length: 24,
      aperture: 4,
      profile: 'Canon EF 24-105mm f/4L IS USM',
    })
    render(<LensNote docId="abc" load={load} />)
    expect(
      await screen.findByText(
        'Using the profile for Canon EF 24-105mm f/4L IS USM at 24 mm.',
      ),
    ).toBeInTheDocument()
    expect(load).toHaveBeenCalledWith('abc', '')
  })

  it('says when there is no profile and what to do', async () => {
    const load = vi.fn().mockResolvedValue({
      camera: null,
      lens: 'Mystery 50mm',
      focal_length: 50,
      aperture: null,
      profile: null,
    })
    render(<LensNote docId="abc" lensName="Mystery 50mm" load={load} />)
    expect(
      await screen.findByText(/no profile for Mystery 50mm/),
    ).toBeInTheDocument()
    expect(load).toHaveBeenCalledWith('abc', 'Mystery 50mm')
  })
})
