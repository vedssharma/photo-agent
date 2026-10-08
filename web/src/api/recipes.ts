import { api } from './client'
import { ApiError, type DocumentView, type Layer } from './documents'
import type { components } from './schema'

export type Recipe = components['schemas']['Recipe']

function fail(response: Response): never {
  throw new ApiError(`The server answered ${response.status}.`)
}

/** Saved recipes, newest first. */
export async function listRecipes(): Promise<Recipe[]> {
  const { data, response } = await api.GET('/api/recipes')
  return data ?? fail(response)
}

/** Save layers as a recipe. Brush masks are left out; they only fit this photo. */
export async function createRecipe(
  name: string,
  layers: Layer[],
): Promise<Recipe> {
  const { data, response } = await api.POST('/api/recipes', {
    body: { name, layers },
  })
  return data ?? fail(response)
}

export async function deleteRecipe(id: string): Promise<void> {
  const { response } = await api.DELETE('/api/recipes/{recipe_id}', {
    params: { path: { recipe_id: id } },
  })
  if (!response.ok) fail(response)
}

/** Add a recipe's layers to a photo, as one step in its history. */
export async function applyRecipe(
  docId: string,
  recipeId: string,
): Promise<DocumentView> {
  const { data, response } = await api.POST(
    '/api/documents/{doc_id}/recipes/{recipe_id}',
    { params: { path: { doc_id: docId, recipe_id: recipeId } } },
  )
  return data ?? fail(response)
}
