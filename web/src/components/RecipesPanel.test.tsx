import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'

import type { Layer } from '../api/documents'
import type { Recipe } from '../api/recipes'
import { makeDoc, stubApi } from '../test/fixtures'
import { RecipesPanel } from './RecipesPanel'

const warm: Layer = {
  id: 'Lwarm',
  name: 'Warmer',
  visible: true,
  opacity: 100,
  blend_mode: 'normal',
  operations: [{ id: 'wb', op: 'white_balance', temperature: 20, tint: 0 }],
}
const moody: Recipe = {
  id: 'r1',
  name: 'Moody',
  created_at: '2026-10-08T00:00:00Z',
  layers: [warm],
}

describe('RecipesPanel', () => {
  it('saves the layers as a named recipe', async () => {
    const doc = makeDoc({ state: { framing: [], layers: [warm] } })
    let sent: unknown
    stubApi({
      'GET /api/recipes': () => Response.json([]),
      'POST /api/recipes': async (req) => {
        sent = await req.json()
        return Response.json({ ...moody, id: 'r2', name: 'Golden' })
      },
    })
    render(<RecipesPanel doc={doc} onDocument={vi.fn()} />)
    await userEvent.click(screen.getByRole('button', { name: 'Save look…' }))
    await userEvent.type(
      screen.getByRole('textbox', { name: 'Recipe name' }),
      'Golden{Enter}',
    )
    expect(
      await screen.findByRole('button', { name: 'Apply “Golden”' }),
    ).toBeInTheDocument()
    expect(sent).toEqual({ name: 'Golden', layers: [warm] })
  })

  it('applies and deletes saved recipes', async () => {
    const doc = makeDoc()
    const applied = makeDoc({ state: { framing: [], layers: [warm] } })
    const onDocument = vi.fn()
    stubApi({
      'GET /api/recipes': () => Response.json([moody]),
      'POST /api/documents/abc123abc123/recipes/r1': () =>
        Response.json(applied),
      'DELETE /api/recipes/r1': () => new Response(null, { status: 204 }),
    })
    render(<RecipesPanel doc={doc} onDocument={onDocument} />)
    expect(screen.getByRole('button', { name: 'Save look…' })).toBeDisabled()
    await userEvent.click(
      await screen.findByRole('button', { name: 'Apply “Moody”' }),
    )
    expect(onDocument).toHaveBeenCalledWith(applied)
    await userEvent.click(
      screen.getByRole('button', { name: 'Delete recipe Moody' }),
    )
    expect(await screen.findByText(/No recipes yet/)).toBeInTheDocument()
  })

  it('adds a built-in look', async () => {
    const doc = makeDoc()
    const applied = makeDoc({ state: { framing: [], layers: [warm] } })
    const onDocument = vi.fn()
    let body: unknown
    stubApi({
      'GET /api/recipes': () => Response.json([]),
      'GET /api/looks': () =>
        Response.json([
          {
            id: 'noir',
            name: 'Noir',
            description: 'Black and white.',
            operations: [{ id: 's', op: 'saturation', amount: -100 }],
          },
        ]),
      'POST /api/documents/abc123abc123/looks/noir': async (req) => {
        body = await req.json()
        return Response.json(applied)
      },
    })
    render(<RecipesPanel doc={doc} onDocument={onDocument} />)
    await userEvent.selectOptions(
      await screen.findByRole('combobox', { name: /Add a look/ }),
      'noir',
    )
    await vi.waitFor(() => expect(onDocument).toHaveBeenCalledWith(applied))
    expect(body).toEqual({ strength: 100 })
  })
})
