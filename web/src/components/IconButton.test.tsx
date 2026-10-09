import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { IconButton } from './IconButton'

describe('IconButton', () => {
  it('names the action and shows a tooltip on hover', async () => {
    const onClick = vi.fn()
    render(
      <IconButton label="Cut out" hint="Keep the subject" onClick={onClick}>
        <svg />
      </IconButton>,
    )
    const button = screen.getByRole('button', { name: 'Cut out' })
    expect(button).toHaveAccessibleDescription('Keep the subject')
    expect(screen.queryByRole('tooltip')).toBeNull()

    await userEvent.hover(button)
    expect(screen.getByRole('tooltip')).toHaveTextContent(
      'Cut outKeep the subject',
    )
    await userEvent.unhover(button)
    expect(screen.queryByRole('tooltip')).toBeNull()

    await userEvent.click(button)
    expect(onClick).toHaveBeenCalledOnce()
  })
})
