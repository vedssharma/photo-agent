import { type FormEvent, useEffect, useState } from 'react'

import { type DocumentView, PHOTO_TYPES } from '../api/documents'
import { matchReference, uploadReference } from '../api/references'
import {
  type Look,
  type Recipe,
  applyLook,
  applyRecipe,
  createRecipe,
  deleteRecipe,
  listLooks,
  listRecipes,
} from '../api/recipes'

interface Props {
  doc: DocumentView
  onDocument: (doc: DocumentView) => void
  /** True while another edit is in flight. */
  disabled?: boolean
}

/** Add a built-in look, save this photo's layers as a reusable recipe, and apply saved
 * recipes to it. */
export function RecipesPanel({ doc, onDocument, disabled = false }: Props) {
  const [recipes, setRecipes] = useState<Recipe[]>([])
  const [looks, setLooks] = useState<Look[]>([])
  const [naming, setNaming] = useState(false)
  const [name, setName] = useState('')
  const [working, setWorking] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let live = true
    listRecipes()
      .then((found) => {
        if (live) setRecipes(found)
      })
      .catch((err: Error) => {
        if (live) setError(`Could not load recipes: ${err.message}`)
      })
    // Without the looks the panel still works; the picker just stays hidden.
    listLooks()
      .then((found) => {
        if (live) setLooks(found)
      })
      .catch(() => {})
    return () => {
      live = false
    }
  }, [])

  async function run(task: () => Promise<void>) {
    setWorking(true)
    setError(null)
    try {
      await task()
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setWorking(false)
    }
  }

  function save(e: FormEvent) {
    e.preventDefault()
    if (!name.trim()) return
    void run(async () => {
      const recipe = await createRecipe(name, doc.state.layers)
      setRecipes((all) => [recipe, ...all])
      setName('')
      setNaming(false)
    })
  }

  const locked = disabled || working
  const hasLayers = doc.state.layers.length > 0
  return (
    <section className="panel recipes-panel" aria-label="Recipes">
      <div className="panel-head">
        <h2>Recipes</h2>
        {!naming && (
          <button
            type="button"
            className="icon"
            disabled={locked || !hasLayers}
            title={
              hasLayers
                ? 'Save these layers to apply to other photos'
                : 'Add a layer first'
            }
            onClick={() => setNaming(true)}
          >
            Save look…
          </button>
        )}
      </div>
      {looks.length > 0 && (
        <label className="look-picker">
          Add a look{' '}
          <select
            value=""
            disabled={locked}
            onChange={(e) => {
              const id = e.target.value
              if (id)
                void run(async () => onDocument(await applyLook(doc.id, id)))
            }}
          >
            <option value="">Choose…</option>
            {looks.map((look) => (
              <option key={look.id} value={look.id} title={look.description}>
                {look.name}
              </option>
            ))}
          </select>
        </label>
      )}
      <label
        className="look-picker"
        title="Make this photo's colors and tone look like another photo's"
      >
        Match a photo{' '}
        <input
          type="file"
          accept={PHOTO_TYPES}
          disabled={locked}
          aria-label="Photo to match"
          onChange={(e) => {
            const file = e.target.files?.[0]
            e.target.value = ''
            if (file)
              void run(async () => {
                const ref = await uploadReference(doc.id, file)
                onDocument(await matchReference(doc.id, ref.id))
              })
          }}
        />
      </label>
      {naming && (
        <form className="recipe-form" onSubmit={save}>
          <input
            aria-label="Recipe name"
            placeholder="Name, e.g. Moody film"
            value={name}
            maxLength={80}
            autoFocus
            onChange={(e) => setName(e.target.value)}
          />
          <button type="submit" disabled={locked || !name.trim()}>
            Save
          </button>
          <button type="button" onClick={() => setNaming(false)}>
            Cancel
          </button>
          <p className="fine-print">
            Saves all {doc.state.layers.length} layer
            {doc.state.layers.length === 1 ? '' : 's'}. Crops and painted masks
            stay with this photo.
          </p>
        </form>
      )}
      {recipes.length === 0 ? (
        <p className="empty">
          No recipes yet. Save a look you like to reuse it on other photos.
        </p>
      ) : (
        <ul className="recipe-list">
          {recipes.map((recipe) => (
            <li key={recipe.id} className="recipe">
              <button
                type="button"
                className="recipe-apply"
                disabled={locked}
                title={`Add ${recipe.layers.map((l) => `“${l.name}”`).join(', ')}`}
                onClick={() =>
                  run(async () =>
                    onDocument(await applyRecipe(doc.id, recipe.id)),
                  )
                }
              >
                Apply “{recipe.name}”
              </button>
              <button
                type="button"
                className="icon"
                aria-label={`Delete recipe ${recipe.name}`}
                disabled={locked}
                onClick={() =>
                  run(async () => {
                    await deleteRecipe(recipe.id)
                    setRecipes((all) => all.filter((r) => r.id !== recipe.id))
                  })
                }
              >
                ×
              </button>
            </li>
          ))}
        </ul>
      )}
      {error && (
        <p className="error" role="alert">
          {error}
        </p>
      )}
    </section>
  )
}
