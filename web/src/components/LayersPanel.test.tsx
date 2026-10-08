import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { EditState, Layer } from '../api/documents'
import type { StateUpdate } from '../hooks/useManualEdit'
import { DEFAULT_MASK_TOOL } from '../lib/masks'
import { SPECS, makeDoc } from '../test/fixtures'
import { LayersPanel } from './LayersPanel'

const warm: Layer = {
  id: 'Lwarm',
  name: 'Warmer',
  visible: true,
  opacity: 100,
  blend_mode: 'normal',
  operations: [{ id: 'wb', op: 'white_balance', temperature: 20, tint: 0 }],
}
const punch: Layer = {
  id: 'Lpunch',
  name: 'Punch',
  visible: true,
  opacity: 100,
  blend_mode: 'luminosity',
  operations: [{ id: 'c', op: 'contrast', amount: 25 }],
}
const doc = makeDoc({
  state: {
    framing: [
      {
        id: 'cr',
        op: 'crop',
        aspect: '4:5',
        left: 0,
        top: 0,
        right: 1,
        bottom: 1,
      },
    ],
    layers: [warm, punch],
  },
})

interface Sent {
  label: string
  coalesce?: string
  state: EditState
}

function setup(selected: string | null = null) {
  const sent: Sent[] = []
  const onEdit = vi.fn(
    async (label: string, update: StateUpdate, coalesce?: string) => {
      sent.push({ label, state: update(doc.state), coalesce })
    },
  )
  const onSelect = vi.fn()
  const onMaskTool = vi.fn()
  render(
    <LayersPanel
      doc={doc}
      onEdit={onEdit}
      selected={selected}
      onSelect={onSelect}
      maskTool={DEFAULT_MASK_TOOL}
      onMaskTool={onMaskTool}
      specs={SPECS}
    />,
  )
  return { sent, onSelect, onMaskTool }
}

describe('LayersPanel', () => {
  it('lists layers top first, then the framing', () => {
    setup()
    const names = screen
      .getAllByRole('listitem')
      .filter((li) => li.classList.contains('layer'))
      .map((li) => within(li).getAllByText(/./)[0].textContent)
    expect(names).toEqual(['Punch', 'Warmer', 'Crop & rotate (1)'])
  })

  it('hides a layer as a named manual step', async () => {
    const { sent } = setup()
    await userEvent.click(screen.getByRole('checkbox', { name: 'Show Warmer' }))
    expect(sent[0].label).toBe('Hide “Warmer”')
    expect(sent[0].state.layers[0].visible).toBe(false)
  })

  it('moves and deletes layers', async () => {
    const { sent } = setup()
    expect(screen.getByRole('button', { name: 'Move Punch up' })).toBeDisabled()
    await userEvent.click(
      screen.getByRole('button', { name: 'Move Warmer up' }),
    )
    expect(sent[0].state.layers.map((l) => l.id)).toEqual(['Lpunch', 'Lwarm'])
    await userEvent.click(screen.getByRole('button', { name: 'Delete Punch' }))
    expect(sent[1].label).toBe('Delete layer “Punch”')
    expect(sent[1].state.layers.map((l) => l.id)).toEqual(['Lwarm'])
  })

  it('edits opacity and blend mode of the selected layer', async () => {
    const { sent, onSelect } = setup('Lwarm')
    await userEvent.click(screen.getByRole('button', { name: 'Punch' }))
    expect(onSelect).toHaveBeenCalledWith('Lpunch')

    const opacity = screen.getByRole('slider', { name: 'Opacity' })
    fireEvent.change(opacity, { target: { value: '60' } })
    fireEvent.pointerUp(opacity)
    expect(sent[0]).toMatchObject({
      label: '“Warmer” opacity 60%',
      coalesce: 'layer:Lwarm:opacity',
    })

    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Blend' }),
      'Brightness only',
    )
    expect(sent[1].state.layers[0].blend_mode).toBe('luminosity')
  })
})

describe('LayersPanel masks', () => {
  it('adds a mask to the selected layer and tunes it', async () => {
    const { sent } = setup('Lwarm')
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Mask' }),
      'Brightness range',
    )
    expect(sent[0].label).toBe('Brightness range mask on “Warmer”')
    expect(sent[0].state.layers[0].mask).toMatchObject({
      kind: 'luminosity',
      low: 0.6,
    })
    expect(sent[0].coalesce).toBeUndefined()
  })

  it('shows controls for the mask a layer has', async () => {
    const masked = makeDoc({
      state: {
        framing: [],
        layers: [
          {
            ...warm,
            mask: { kind: 'brush', strokes: [], invert: false },
          },
        ],
      },
    })
    const onEdit = vi.fn(async () => {})
    const onMaskTool = vi.fn()
    render(
      <LayersPanel
        doc={masked}
        onEdit={onEdit}
        selected="Lwarm"
        onSelect={vi.fn()}
        maskTool={DEFAULT_MASK_TOOL}
        onMaskTool={onMaskTool}
        specs={SPECS}
      />,
    )
    expect(screen.getByText(/Paint on the photo/)).toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'Erase' }))
    expect(onMaskTool).toHaveBeenCalledWith({
      ...DEFAULT_MASK_TOOL,
      erase: true,
    })
    await userEvent.click(screen.getByRole('checkbox', { name: 'Invert' }))
    expect(onEdit).toHaveBeenCalledWith(
      '“Warmer” mask inverted',
      expect.any(Function),
      'layer:Lwarm:mask:invert',
    )
  })
})

describe('LayersPanel manual controls', () => {
  it('tunes an operation with its slider', () => {
    const { sent } = setup('Lwarm')
    const temperature = screen.getByRole('slider', { name: 'Temperature' })
    expect(temperature).toHaveValue('20')
    fireEvent.change(temperature, { target: { value: '35' } })
    fireEvent.pointerUp(temperature)
    expect(sent[0]).toMatchObject({
      label: 'White balance: temperature +35',
      coalesce: 'op:wb:temperature',
    })
    expect(sent[0].state.layers[0].operations[0]).toMatchObject({
      temperature: 35,
      tint: 0,
    })
  })

  it('adds and removes adjustments', async () => {
    const { sent } = setup('Lwarm')
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Add adjustment' }),
      'Exposure',
    )
    expect(sent[0].label).toBe('Add Exposure to “Warmer”')
    expect(sent[0].state.layers[0].operations[1]).toMatchObject({
      op: 'exposure',
      stops: 0,
    })
    await userEvent.click(
      screen.getByRole('button', { name: 'Remove White balance' }),
    )
    expect(sent[1].state.layers[0].operations).toEqual([])
  })

  it('edits the framing', async () => {
    const { sent, onSelect } = setup('framing')
    expect(screen.getByRole('slider', { name: 'Left' })).toBeInTheDocument()
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Aspect' }),
      '1:1',
    )
    expect(sent[0].state.framing[0]).toMatchObject({ aspect: '1:1' })
    await userEvent.click(screen.getByRole('button', { name: /Crop & rotate/ }))
    expect(onSelect).toHaveBeenCalledWith(null)
  })

  it('creates a new empty layer and selects it', async () => {
    const { sent, onSelect } = setup()
    await userEvent.click(screen.getByRole('button', { name: '+ New layer' }))
    expect(sent[0].label).toBe('New layer')
    const added = sent[0].state.layers[2]
    expect(added).toMatchObject({ name: 'Layer 3', operations: [] })
    expect(onSelect).toHaveBeenCalledWith(added.id)
  })
})
