import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { EditState, Layer } from '../api/documents'
import type { StateUpdate } from '../hooks/useManualEdit'
import { DEFAULT_MASK_TOOL, semanticMask } from '../lib/masks'
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
  const onRetouch = vi.fn()
  const onStraighten = vi.fn()
  render(
    <LayersPanel
      doc={doc}
      onEdit={onEdit}
      selected={selected}
      onSelect={onSelect}
      maskTool={DEFAULT_MASK_TOOL}
      onMaskTool={onMaskTool}
      specs={SPECS}
      onRetouch={onRetouch}
      onStraighten={onStraighten}
    />,
  )
  return { sent, onSelect, onMaskTool, onRetouch, onStraighten }
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
  it('selects the sky with AI', async () => {
    const { sent } = setup('Lwarm')
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Mask' }),
      'Sky',
    )
    expect(sent[0].label).toBe('Select sky for “Warmer”')
    expect(sent[0].state.layers[0].mask).toEqual({
      kind: 'semantic',
      target: 'sky',
      points: [],
      strokes: [],
      description: '',
      invert: false,
    })
  })

  it('waits for a click on the photo to select an object', async () => {
    const { sent, onMaskTool } = setup('Lwarm')
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Mask' }),
      'An object (click it)',
    )
    expect(sent).toEqual([])
    expect(onMaskTool).toHaveBeenLastCalledWith({
      ...DEFAULT_MASK_TOOL,
      picking: true,
    })
  })

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

describe('LayersPanel touch-ups', () => {
  const sky = makeDoc({
    state: {
      framing: [],
      layers: [{ ...warm, mask: semanticMask('sky') }],
    },
  })

  function renderSky(tool = DEFAULT_MASK_TOOL) {
    const onEdit = vi.fn(async () => {})
    const onMaskTool = vi.fn()
    render(
      <LayersPanel
        doc={sky}
        onEdit={onEdit}
        selected="Lwarm"
        onSelect={vi.fn()}
        maskTool={tool}
        onMaskTool={onMaskTool}
        specs={SPECS}
      />,
    )
    return { onEdit, onMaskTool }
  }

  it('turns on the brush for touching up an AI selection', async () => {
    const { onMaskTool } = renderSky()
    expect(screen.queryByRole('slider', { name: 'Brush size' })).toBeNull()
    await userEvent.click(
      screen.getByRole('checkbox', { name: /Touch up with brush/ }),
    )
    expect(onMaskTool).toHaveBeenCalledWith({
      ...DEFAULT_MASK_TOOL,
      refining: true,
    })
  })

  it('shows brush controls while touching up', () => {
    renderSky({ ...DEFAULT_MASK_TOOL, refining: true })
    expect(screen.getByText(/add what the AI missed/)).toBeInTheDocument()
    expect(screen.getByRole('slider', { name: 'Brush size' })).toBeVisible()
    expect(
      screen.getByRole('button', { name: 'Clear touch-ups' }),
    ).toBeDisabled()
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
    await userEvent.click(screen.getByRole('button', { name: 'New layer' }))
    expect(sent[0].label).toBe('New layer')
    const added = sent[0].state.layers[2]
    expect(added).toMatchObject({ name: 'Layer 3', operations: [] })
    expect(onSelect).toHaveBeenCalledWith(added.id)
  })
})

describe('LayersPanel removal', () => {
  it('adds a removal layer and waits for a click on what to remove', async () => {
    const { sent, onSelect, onMaskTool } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Remove' }))
    expect(sent[0].label).toBe('Remove an object')
    const added = sent[0].state.layers[2]
    expect(added).toMatchObject({ name: 'Remove object', mask: null })
    expect(added.operations).toMatchObject([{ op: 'remove', grow: 20 }])
    expect(onSelect).toHaveBeenCalledWith(added.id)
    expect(onMaskTool).toHaveBeenCalledWith({
      ...DEFAULT_MASK_TOOL,
      picking: true,
    })
  })
})

describe('LayersPanel cutout', () => {
  it('removes the background, keeping the main subject', async () => {
    const { sent, onSelect } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Cut out' }))
    expect(sent[0].label).toBe('Remove the background')
    expect(sent[0].state.cutout).toMatchObject({
      visible: true,
      background: null,
      mask: { kind: 'semantic', target: 'subject' },
    })
    expect(onSelect).toHaveBeenCalledWith('cutout')
  })
})

describe('LayersPanel retouch', () => {
  it('asks for the portrait retouch layers', async () => {
    const { sent, onRetouch } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Retouch' }))
    expect(onRetouch).toHaveBeenCalledOnce()
    expect(sent).toEqual([])
  })
})

describe('LayersPanel framing', () => {
  it('straightens automatically from the framing section', async () => {
    const { sent, onStraighten } = setup('framing')
    await userEvent.click(
      screen.getByRole('button', { name: 'Auto straighten' }),
    )
    expect(onStraighten).toHaveBeenCalledOnce()
    expect(sent).toEqual([])
  })
})

describe('LayersPanel generative fill', () => {
  it('asks what to add, then adds a layer to paint it into', async () => {
    const { sent, onSelect } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Generate' }))
    await userEvent.type(
      screen.getByRole('textbox', { name: 'What to add' }),
      'a potted fern{Enter}',
    )
    expect(sent[0].label).toBe('Generate “a potted fern”')
    const added = sent[0].state.layers.at(-1)
    expect(added?.name).toBe('A potted fern')
    expect(added?.mask).toEqual({ kind: 'brush', strokes: [], invert: false })
    expect(added?.operations[0]).toMatchObject({
      op: 'generate',
      prompt: 'a potted fern',
    })
    expect(onSelect).toHaveBeenCalledWith(added?.id)
    expect(screen.queryByRole('textbox', { name: 'What to add' })).toBeNull()
  })
})

describe('LayersPanel expand canvas', () => {
  it('expands the framing to an aspect ratio', async () => {
    const { sent } = setup('framing')
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Expand' }),
      '16:9',
    )
    expect(sent[0].label).toBe('Expand canvas to 16:9')
    expect(sent[0].state.framing.at(-1)).toMatchObject({
      op: 'expand',
      aspect: '16:9',
      prompt: '',
    })
  })
})

describe('LayersPanel new background', () => {
  it('replaces everything but the subject with a described scene', async () => {
    const { sent } = setup()
    await userEvent.click(
      screen.getByRole('button', { name: 'New background' }),
    )
    await userEvent.type(
      screen.getByRole('textbox', { name: 'New background' }),
      'a sunlit beach{Enter}',
    )
    expect(sent[0].label).toBe('New background: “a sunlit beach”')
    const added = sent[0].state.layers.at(-1)
    expect(added?.name).toBe('Background: A sunlit beach')
    expect(added?.mask).toBeNull()
    expect(added?.operations[0]).toMatchObject({
      op: 'replace_background',
      prompt: 'a sunlit beach',
      harmonize: 50,
    })
  })

  it('relights from the chosen direction in a layer of its own', async () => {
    const { sent } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Relight' }))
    await userEvent.selectOptions(
      screen.getByRole('combobox', { name: 'Where the light comes from' }),
      'top',
    )
    await userEvent.type(
      screen.getByRole('textbox', { name: 'The new light' }),
      'soft window light{Enter}',
    )
    expect(sent[0].label).toBe('Relight: “soft window light”')
    const added = sent[0].state.layers.at(-1)
    expect(added?.name).toBe('Light: Soft window light')
    expect(added?.mask).toBeNull()
    expect(added?.operations[0]).toMatchObject({
      op: 'relight',
      direction: 'top',
      prompt: 'soft window light',
      amount: 60,
    })
  })

  it('restores faces and colorizes in layers of their own', async () => {
    const { sent } = setup()
    await userEvent.click(screen.getByRole('button', { name: 'Restore' }))
    await userEvent.click(
      screen.getByRole('button', { name: 'Colorize black and white' }),
    )
    expect(sent[0].label).toBe('Colorize')
    const added = sent[0].state.layers.at(-1)
    expect(added?.mask).toBeNull()
    expect(added?.operations[0]).toMatchObject({ op: 'colorize', amount: 100 })
  })
})
