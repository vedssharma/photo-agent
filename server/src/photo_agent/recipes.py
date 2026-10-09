"""Edit recipes: a set of layers saved under a name, to apply to other photos.

A recipe keeps the look (adjustment layers with their opacity, blend mode, and masks) and
leaves out what only fits the photo it was made on: framing, brush masks, which were
painted on that photo's content, and selections of one particular object. Gradient and
brightness-range masks carry over, since they are relative to the frame or to the photo's
own tones, and so do selections like "the sky" or "skin", which are found anew in each photo.
Layers that remove or generate something in a place that only this photo has are left out.

Recipes live in one JSON file, `<data_dir>/recipes.json`, shared by every document.
"""

from __future__ import annotations

import threading
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field, TypeAdapter

from photo_agent.layers import EditState, Layer, new_layer_id
from photo_agent.masks import BrushMask, Mask, SemanticMask
from photo_agent.operations import new_op_id

MAX_RECIPES = 200


class RecipeNotFoundError(KeyError):
    pass


class Recipe(BaseModel):
    id: str
    name: str = Field(min_length=1, max_length=80)
    created_at: datetime
    layers: list[Layer] = Field(min_length=1)


def portable(layers: Sequence[Layer]) -> list[Layer]:
    """Copies of the layers that make sense on any photo (photo-specific masks dropped)."""
    out = []
    for layer in layers:
        copy = layer.model_copy(deep=True)
        if copy.mask is not None and not is_portable(copy.mask):
            if copy.is_content:
                continue  # removing or generating "there" means nothing on another photo
            copy.mask = None
        if isinstance(copy.mask, SemanticMask):
            copy.mask.strokes = []  # painted for this photo
        out.append(copy)
    return out


def is_portable(mask: Mask) -> bool:
    if isinstance(mask, BrushMask):
        return False
    return not (isinstance(mask, SemanticMask) and mask.target == "object")


def apply(recipe: Recipe, state: EditState) -> EditState:
    """The state with the recipe's layers added on top, each with fresh ids."""
    added = []
    for layer in recipe.layers:
        copy = layer.model_copy(deep=True, update={"id": new_layer_id()})
        for op in copy.operations:
            op.id = new_op_id()
        added.append(copy)
    return state.model_copy(update={"layers": [*state.layers, *added]}, deep=True)


_RECIPES = TypeAdapter(list[Recipe])


class RecipeStore:
    def __init__(self, root: Path) -> None:
        self.path = root / "recipes.json"
        self._lock = threading.Lock()

    def all(self) -> list[Recipe]:
        """Newest first."""
        if not self.path.is_file():
            return []
        return _RECIPES.validate_json(self.path.read_bytes())

    def get(self, recipe_id: str) -> Recipe:
        for recipe in self.all():
            if recipe.id == recipe_id:
                return recipe
        raise RecipeNotFoundError(recipe_id)

    def create(self, name: str, layers: Sequence[Layer]) -> Recipe:
        recipe = Recipe(
            id=uuid.uuid4().hex[:12],
            name=name.strip(),
            created_at=datetime.now(UTC),
            layers=portable(layers),
        )
        with self._lock:
            self._write([recipe, *self.all()][:MAX_RECIPES])
        return recipe

    def delete(self, recipe_id: str) -> None:
        with self._lock:
            recipes = self.all()
            kept = [r for r in recipes if r.id != recipe_id]
            if len(kept) == len(recipes):
                raise RecipeNotFoundError(recipe_id)
            self._write(kept)

    def _write(self, recipes: list[Recipe]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_bytes(_RECIPES.dump_json(recipes, indent=2))
        tmp.replace(self.path)
