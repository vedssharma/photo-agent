"""Open-source model backends, loaded from the Hugging Face Hub on first use.

These need the optional `models` extra (`uv sync --extra models`); weights download once
into the model directory and stay loaded in the worker between jobs. Each backend records
its weights' license, so moving to a product later does not mean swapping models blind.

    sam2.1-hiera-small      facebook/sam2.1-hiera-small        Apache-2.0
    upernet-convnext-ade    openmmlab/upernet-convnext-small   MIT
    birefnet                ZhengPeng7/BiRefNet                MIT
    segformer-face-parsing  jonathandinu/face-parsing          unspecified; trained on
                                                               CelebAMask-HQ (non-commercial)
"""

from __future__ import annotations

import threading
from typing import Any, ClassVar, cast

import cv2
import numpy as np

from photo_agent.imaging import Array
from photo_agent.vision import classical
from photo_agent.vision.backends import Env, Progress, has_packages

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)


class HubBackend:
    """Shared loading and caching. Subclasses set the names and implement `_load`/`run`."""

    name: str = ""
    license: str = ""
    repo: str = ""
    uses_weights = True
    packages: tuple[str, ...] = ("torch", "transformers")
    _loaded: ClassVar[dict[str, Any]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    def available(self) -> bool:
        return has_packages(*self.packages)

    def model(self, env: Env, progress: Progress) -> Any:
        """The loaded model, downloading its weights the first time."""
        with self._lock:
            key = f"{self.name}@{env.device}"
            if key not in self._loaded:
                progress(None, "Loading the model")
                self._loaded[key] = self._load(env)
            return self._loaded[key]

    def _load(self, env: Env) -> Any:
        raise NotImplementedError

    def _cache(self, env: Env) -> str:
        return str(env.cache_dir / "huggingface")


def _to_pil(image: Array) -> Any:
    from PIL import Image

    return Image.fromarray((np.clip(image, 0, 1) * 255 + 0.5).astype(np.uint8))


def _upsample(prob: Any, shape: tuple[int, int]) -> Array:
    m = np.asarray(prob, np.float32)
    if m.shape != shape:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return cast(Array, np.clip(m, 0.0, 1.0).astype(np.float32))


class Sam2Object(HubBackend):
    """Segment Anything 2: one object from a box and/or points."""

    name = "sam2.1-hiera-small"
    license = "Apache-2.0"
    repo = "facebook/sam2.1-hiera-small"

    def _load(self, env: Env) -> Any:
        from transformers import Sam2Model, Sam2Processor

        processor = Sam2Processor.from_pretrained(self.repo, cache_dir=self._cache(env))
        model = Sam2Model.from_pretrained(self.repo, cache_dir=self._cache(env))
        return processor, model.to(env.device).eval()

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch

        processor, model = self.model(env, progress)
        h, w = image.shape[:2]
        kwargs: dict[str, Any] = {}
        box = params.get("box")
        if box is not None:
            kwargs["input_boxes"] = [[[box[0] * w, box[1] * h, box[2] * w, box[3] * h]]]
        points = params.get("points") or []
        if points:
            kwargs["input_points"] = [[[[p[0] * w, p[1] * h] for p in points]]]
            kwargs["input_labels"] = [[[1 if p[2] else 0 for p in points]]]
        progress(0.3, "Selecting")
        inputs = processor(images=_to_pil(image), return_tensors="pt", **kwargs).to(env.device)
        with torch.inference_mode():
            outputs = model(**inputs, multimask_output=False)
        masks = processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"], binarize=False
        )[0]
        logits = masks[0, 0].float().numpy()
        return classical.finish(1.0 / (1.0 + np.exp(-logits)), image, soften=0.002)


class AdeSegmenter(HubBackend):
    """UperNet (ConvNeXt) trained on ADE20K scene parsing: the sky, or people."""

    name = "upernet-convnext-ade"
    license = "MIT"
    repo = "openmmlab/upernet-convnext-small"
    LABELS: ClassVar[dict[str, int]] = {"sky": 2, "people": 12}

    def __init__(self, label: str) -> None:
        self.label = label

    def _load(self, env: Env) -> Any:
        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

        processor = AutoImageProcessor.from_pretrained(self.repo, cache_dir=self._cache(env))
        model = UperNetForSemanticSegmentation.from_pretrained(
            self.repo, cache_dir=self._cache(env)
        )
        return processor, model.to(env.device).eval()

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch

        processor, model = self.model(env, progress)
        progress(0.3, "Looking at the scene")
        inputs = processor(images=_to_pil(image), return_tensors="pt").to(env.device)
        with torch.inference_mode():
            logits = model(**inputs).logits
        probs = torch.softmax(logits[0].float(), dim=0)[self.LABELS[self.label]].cpu().numpy()
        return classical.finish(_upsample(probs, image.shape[:2]), image)


class BiRefNetSubject(HubBackend):
    """BiRefNet: a high-resolution matte of the main subject, as for a cutout."""

    name = "birefnet"
    license = "MIT"
    repo = "ZhengPeng7/BiRefNet"
    packages = ("torch", "transformers", "timm", "kornia", "einops")
    SIZE = 1024

    def _load(self, env: Env) -> Any:
        from transformers import AutoModelForImageSegmentation

        model = AutoModelForImageSegmentation.from_pretrained(
            self.repo, trust_remote_code=True, cache_dir=self._cache(env)
        )
        return model.to(env.device).eval()

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch

        model = self.model(env, progress)
        progress(0.3, "Finding the subject")
        x = cv2.resize(image, (self.SIZE, self.SIZE), interpolation=cv2.INTER_AREA)
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)[None].copy()).to(env.device)
        with torch.inference_mode():
            pred = model(tensor)[-1].sigmoid()[0, 0].float().cpu().numpy()
        # A matte already has soft, accurate edges; only scale it back.
        return _upsample(pred, image.shape[:2])


class FaceParser(HubBackend):
    """SegFormer face parsing (CelebAMask-HQ labels), run on each face found."""

    name = "segformer-face-parsing"
    license = "unspecified (trained on CelebAMask-HQ, non-commercial)"
    repo = "jonathandinu/face-parsing"
    PARTS: ClassVar[dict[str, tuple[int, ...]]] = {
        "skin": (1, 2, 8, 9, 17),
        "face": (1, 2, 4, 5, 6, 7, 10, 11, 12),
        "eyes": (4, 5),
        "lips": (11, 12),
        "mouth": (10,),
        "hair": (13,),
    }

    def _load(self, env: Env) -> Any:
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        processor = SegformerImageProcessor.from_pretrained(self.repo, cache_dir=self._cache(env))
        model = SegformerForSemanticSegmentation.from_pretrained(
            self.repo, cache_dir=self._cache(env)
        )
        return processor, model.to(env.device).eval()

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch

        processor, model = self.model(env, progress)
        h, w = image.shape[:2]
        faces = classical.find_faces(image) or [(0.0, 0.0, float(w), float(h))]
        labels = np.zeros((h, w), np.uint8)
        for i, (x0, y0, x1, y1) in enumerate(faces):
            progress(0.2 + 0.7 * i / len(faces), "Finding faces")
            # Parse a generous crop around each face: the model was trained on portraits.
            pad_x, pad_y = (x1 - x0) * 0.6, (y1 - y0) * 0.6
            cx0, cy0 = max(0, int(x0 - pad_x)), max(0, int(y0 - pad_y))
            cx1, cy1 = min(w, int(x1 + pad_x)), min(h, int(y1 + pad_y * 1.5))
            crop = image[cy0:cy1, cx0:cx1]
            inputs = processor(images=_to_pil(crop), return_tensors="pt").to(env.device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            up = torch.nn.functional.interpolate(
                logits, size=crop.shape[:2], mode="bilinear", align_corners=False
            )
            found = up.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
            region = labels[cy0:cy1, cx0:cx1]
            labels[cy0:cy1, cx0:cx1] = np.where(found > 0, found, region)
        parts = {name: np.isin(labels, ids).astype(np.float32) for name, ids in self.PARTS.items()}
        hsv = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2HSV)
        mouth = parts.pop("mouth").astype(bool)
        parts["teeth"] = (mouth & (hsv[..., 2] > 0.5) & (hsv[..., 1] < 0.35)).astype(np.float32)
        return {name: classical.finish(m, image, soften=0.0015) for name, m in parts.items()}
