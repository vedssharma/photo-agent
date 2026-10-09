import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { Operation } from '../api/documents'
import type { OperationSpec } from '../api/operations'
import { OperationControls } from './OperationControls'

const spec: OperationSpec = {
  op: 'generate',
  label: 'Generate',
  description: '',
  group: 'generative',
  framing: false,
  params: [
    { name: 'prompt', label: 'Prompt', kind: 'text', description: '' },
    { name: 'seed', label: 'Seed', kind: 'seed', description: '' },
  ],
}

const op = {
  id: 'g1',
  op: 'generate',
  prompt: 'a fern',
  seed: 42,
  grow: 10,
  model: 'classical',
} as Operation

describe('OperationControls for generative edits', () => {
  it('changes the prompt when the person presses Enter', async () => {
    const onChange = vi.fn()
    render(
      <OperationControls
        op={op}
        spec={spec}
        onChange={onChange}
        onRemove={() => {}}
      />,
    )
    const prompt = screen.getByRole('textbox', { name: 'Prompt' })
    await userEvent.clear(prompt)
    await userEvent.type(prompt, 'a cactus{Enter}')
    expect(onChange).toHaveBeenCalledWith(
      { prompt: 'a cactus' },
      'Generate: “a cactus”',
      'prompt',
    )
  })

  it('shows the seed and asks for a new take', async () => {
    const onChange = vi.fn()
    render(
      <OperationControls
        op={op}
        spec={spec}
        onChange={onChange}
        onRemove={() => {}}
      />,
    )
    expect(screen.getByText('42')).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'New take' }))
    const [changes, label] = onChange.mock.calls[0] as [
      { seed: number },
      string,
    ]
    expect(label).toBe('Generate: new take')
    expect(changes.seed).not.toBe(42)
    expect(changes.seed).toBeGreaterThanOrEqual(0)
  })
})
