import { useEffect, useState } from 'react'

import type { BatchRequest, LibraryPhoto } from '../../api/library'
import { type Recipe, listRecipes } from '../../api/recipes'

type Source = 'describe' | 'copy' | 'recipe'

interface Props {
  /** The selected photos; the first is the reference unless another is chosen. */
  photos: LibraryPhoto[]
  onStart: (request: BatchRequest) => void
  onClose: () => void
}

/** Edit many photos together: one look, with each evened out to match the set. */
export function BatchDialog({ photos, onStart, onClose }: Props) {
  const [source, setSource] = useState<Source>('describe')
  const [instruction, setInstruction] = useState('')
  const [reference, setReference] = useState(photos[0]?.id ?? '')
  const [recipes, setRecipes] = useState<Recipe[]>([])
  const [recipe, setRecipe] = useState('')
  const [match, setMatch] = useState(60)

  useEffect(() => {
    let live = true
    listRecipes().then(
      (found) => live && setRecipes(found),
      () => {},
    )
    return () => {
      live = false
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const chosenRecipe = recipe || recipes[0]?.id || ''
  const ready =
    source === 'describe'
      ? instruction.trim().length > 0
      : source === 'recipe'
        ? !!chosenRecipe
        : true

  function start() {
    const ids = photos.map((p) => p.id)
    onStart({
      ids,
      reference,
      match,
      instruction: source === 'describe' ? instruction.trim() : null,
      recipe: source === 'recipe' ? chosenRecipe : null,
    })
  }

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <form
        className="modal batch-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="batch-title"
        onClick={(e) => e.stopPropagation()}
        onSubmit={(e) => {
          e.preventDefault()
          if (ready) start()
        }}
      >
        <h2 id="batch-title">Edit {photos.length} photos together</h2>
        <fieldset>
          <legend>The look</legend>
          <label>
            <input
              type="radio"
              name="source"
              checked={source === 'describe'}
              onChange={() => setSource('describe')}
            />
            Describe it
          </label>
          {source === 'describe' && (
            <textarea
              aria-label="Describe the look"
              rows={3}
              value={instruction}
              placeholder="Make all of these consistent: warm, bright, a little faded"
              onChange={(e) => setInstruction(e.target.value)}
            />
          )}
          <label>
            <input
              type="radio"
              name="source"
              checked={source === 'copy'}
              onChange={() => setSource('copy')}
            />
            Copy the reference photo’s edits
          </label>
          <label>
            <input
              type="radio"
              name="source"
              checked={source === 'recipe'}
              disabled={recipes.length === 0}
              onChange={() => setSource('recipe')}
            />
            Use a recipe{recipes.length === 0 && ' (none saved yet)'}
          </label>
          {source === 'recipe' && (
            <select
              aria-label="Recipe"
              value={chosenRecipe}
              onChange={(e) => setRecipe(e.target.value)}
            >
              {recipes.map((r) => (
                <option key={r.id} value={r.id}>
                  {r.name}
                </option>
              ))}
            </select>
          )}
        </fieldset>
        <label className="field">
          Reference photo
          <select
            value={reference}
            onChange={(e) => setReference(e.target.value)}
          >
            {photos.map((p) => (
              <option key={p.id} value={p.id}>
                {p.filename}
              </option>
            ))}
          </select>
        </label>
        <p className="hint">
          {source === 'describe'
            ? 'Claude edits the reference photo first; the others follow its look.'
            : 'The others follow its look.'}
        </p>
        <label className="field">
          Even out each photo: {match}%
          <input
            type="range"
            min={0}
            max={100}
            step={10}
            value={match}
            onChange={(e) => setMatch(Number(e.target.value))}
          />
        </label>
        <p className="hint">
          Moves each photo’s color and brightness toward the reference, so shots
          from different light match. Every photo can be undone on its own.
        </p>
        <div className="modal-actions">
          <button type="button" onClick={onClose}>
            Cancel
          </button>
          <button type="submit" className="primary" disabled={!ready}>
            Edit {photos.length} photos
          </button>
        </div>
      </form>
    </div>
  )
}
