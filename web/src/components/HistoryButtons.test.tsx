import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import { makeDoc, stubApi } from '../test/fixtures'
import { HistoryButtons } from './HistoryButtons'

describe('HistoryButtons', () => {
  it('disables what is not possible', () => {
    render(<HistoryButtons doc={makeDoc()} onDocument={vi.fn()} />)
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled()
    expect(screen.getByRole('button', { name: 'Redo' })).toBeDisabled()
  })

  it('undoes the last agent turn', async () => {
    const undone = makeDoc({ can_redo: true, redo_label: 'request t1' })
    const fetchMock = stubApi({
      'POST /api/documents/abc123abc123/undo': () => Response.json(undone),
    })
    const onDocument = vi.fn()
    const doc = makeDoc({ can_undo: true, undo_label: 'request t1' })
    render(<HistoryButtons doc={doc} onDocument={onDocument} />)

    const button = screen.getByRole('button', { name: 'Undo' })
    expect(button).toHaveAttribute(
      'title',
      expect.stringContaining('request t1'),
    )
    await userEvent.click(button)

    expect(onDocument).toHaveBeenCalledWith(undone)
    expect(fetchMock).toHaveBeenCalledOnce()
  })

  it('redoes with the keyboard shortcut', async () => {
    stubApi({
      'POST /api/documents/abc123abc123/redo': () => Response.json(makeDoc()),
    })
    const onDocument = vi.fn()
    render(
      <HistoryButtons
        doc={makeDoc({ can_redo: true, redo_label: 'request t1' })}
        onDocument={onDocument}
      />,
    )
    fireEvent.keyDown(window, {
      key: 'z',
      ctrlKey: true,
      metaKey: true,
      shiftKey: true,
    })
    await waitFor(() => expect(onDocument).toHaveBeenCalled())
  })

  it('is locked while the agent is editing', () => {
    render(
      <HistoryButtons
        doc={makeDoc({ can_undo: true })}
        onDocument={vi.fn()}
        disabled
      />,
    )
    expect(screen.getByRole('button', { name: 'Undo' })).toBeDisabled()
  })
})
