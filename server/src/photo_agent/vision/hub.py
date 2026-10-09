"""Open-source model backends, loaded from the Hugging Face Hub on first use.

These need the optional `models` extra (`uv sync --extra models`); weights download once
into the model directory and stay loaded in the worker between jobs. Each backend records
its weights' license, so moving to a product later does not mean swapping models blind.

    sam2.1-hiera-small      facebook/sam2.1-hiera-small        Apache-2.0
    upernet-convnext-ade    openmmlab/upernet-convnext-small   MIT
    birefnet                ZhengPeng7/BiRefNet                MIT
    segformer-face-parsing  jonathandinu/face-parsing          unspecified; trained on
                                                               CelebAMask-HQ (non-commercial)
    lama                    Carve/LaMa-ONNX (lama_fp32.onnx)   Apache-2.0
    sdxl-inpainting-0.1     diffusers/stable-diffusion-xl-     CreativeML Open RAIL++-M
                            1.0-inpainting-0.1
    ic-light-fc             lllyasviel/ic-light                CreativeML Open RAIL-M
                            (iclight_sd15_fc) on stablediffusionapi/realistic-vision-v51
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


class LamaInpaint(HubBackend):
    """LaMa (big-lama, ONNX export): fills large holes with plausible texture and structure."""

    name = "lama"
    license = "Apache-2.0"
    repo = "Carve/LaMa-ONNX"
    packages = ("onnxruntime", "huggingface_hub")
    SIZE = 512
    """The export takes a fixed 512x512 input."""

    def _load(self, env: Env) -> Any:
        import onnxruntime
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(self.repo, "lama_fp32.onnx", cache_dir=self._cache(env))
        providers = ["CPUExecutionProvider"]
        if env.device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        return onnxruntime.InferenceSession(path, providers=providers)

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        session = self.model(env, progress)
        hole = np.asarray(params["mask"]) > 0.5
        h, w = hole.shape
        progress(0.3, "Filling in")
        size = (self.SIZE, self.SIZE)
        img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        small = cv2.resize(hole.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        small = cv2.dilate(small, np.ones((5, 5), np.uint8)).astype(np.float32)
        names = [i.name for i in session.get_inputs()]
        mask_name = next((n for n in names if "mask" in n.lower()), names[1])
        image_name = next(n for n in names if n != mask_name)
        feeds = {
            image_name: img.transpose(2, 0, 1)[None].astype(np.float32),
            mask_name: small[None, None],
        }
        out = np.asarray(session.run(None, feeds)[0][0], np.float32).transpose(1, 2, 0)
        if out.max() > 2.0:
            out = out / 255.0
        filled = cv2.resize(np.clip(out, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
        return np.where(hole[..., None], np.clip(filled, 0, 1), image).astype(np.float32)


class SdxlInpaint(HubBackend):
    """Stable Diffusion XL inpainting: paints what `prompt` describes where `mask` is set,
    matched to the light and perspective around it. Without a mask it repaints the whole
    image, keeping as much of it as `strength` (0..1) leaves alone."""

    name = "sdxl-inpainting-0.1"
    license = "CreativeML Open RAIL++-M"
    repo = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
    packages = ("torch", "diffusers", "transformers", "accelerate")
    SIZE = 1024
    """Long edge it generates at: SDXL's native resolution."""
    STEPS = 30
    NEGATIVE = "blurry, low quality, distorted, deformed, watermark, text, frame, border"

    def _load(self, env: Env) -> Any:
        import torch
        from diffusers import AutoPipelineForInpainting

        half = env.device in ("cuda", "mps")
        kwargs: dict[str, Any] = {
            "cache_dir": self._cache(env),
            "torch_dtype": torch.float16 if half else torch.float32,
        }
        if half:
            kwargs["variant"] = "fp16"
        pipe = AutoPipelineForInpainting.from_pretrained(self.repo, **kwargs).to(env.device)
        pipe.set_progress_bar_config(disable=True)
        if env.device == "cpu":
            pipe.enable_attention_slicing()
        return pipe

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch
        from PIL import Image

        pipe = self.model(env, progress)
        h, w = image.shape[:2]
        scale = self.SIZE / max(h, w)
        gw, gh = (max(256, round(v * scale / 8) * 8) for v in (w, h))
        mask = params.get("mask")
        strength = float(params.get("strength", 1.0))
        if mask is None:
            mask = np.ones((h, w), np.float32)
        m = cv2.resize(np.asarray(mask, np.float32), (gw, gh), interpolation=cv2.INTER_LINEAR)
        mask_image = Image.fromarray((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8), "L")
        steps = self.STEPS
        runs = max(1, int(steps * min(strength, 0.99)))

        def on_step(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            progress(0.1 + 0.85 * (step + 1) / runs, "Generating")
            return kwargs

        progress(0.1, "Generating")
        result = pipe(
            prompt=str(params.get("prompt") or ""),
            negative_prompt=str(params.get("negative") or self.NEGATIVE),
            image=_to_pil(cv2.resize(image, (gw, gh), interpolation=cv2.INTER_AREA)),
            mask_image=mask_image,
            width=gw,
            height=gh,
            strength=min(strength, 0.99),
            num_inference_steps=steps,
            guidance_scale=float(params.get("guidance", 7.0)),
            generator=torch.Generator(device="cpu").manual_seed(int(params.get("seed", 0))),
            callback_on_step_end=on_step,
        ).images[0]
        out = (np.asarray(result.convert("RGB")) / 255.0).astype(np.float32)
        return _resize(out, (h, w))


class IcLight(HubBackend):
    """IC-Light (foreground-conditioned): Stable Diffusion 1.5 retrained to redraw a photo
    under new light, keeping its content. The light's look comes from `prompt`; its
    direction from a gradient the diffusion starts from (none for light from the front)."""

    name = "ic-light-fc"
    license = "CreativeML Open RAIL-M"
    repo = "lllyasviel/ic-light"
    base = "stablediffusionapi/realistic-vision-v51"
    weights = "iclight_sd15_fc.safetensors"
    packages = ("torch", "diffusers", "transformers", "safetensors")
    SIZE = 768
    STEPS = 25
    DENOISE = 0.9
    """How much of the directional gradient the diffusion redraws."""
    QUALITY = "best quality"
    NEGATIVE = "lowres, bad anatomy, bad hands, cropped, worst quality"

    def _load(self, env: Env) -> Any:
        import torch
        from diffusers import (
            AutoencoderKL,
            DPMSolverMultistepScheduler,
            StableDiffusionImg2ImgPipeline,
            StableDiffusionPipeline,
            UNet2DConditionModel,
        )
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import CLIPTextModel, CLIPTokenizer

        cache = self._cache(env)
        dtype = torch.float16 if env.device in ("cuda", "mps") else torch.float32
        tokenizer = CLIPTokenizer.from_pretrained(self.base, subfolder="tokenizer", cache_dir=cache)
        text_encoder = CLIPTextModel.from_pretrained(
            self.base, subfolder="text_encoder", cache_dir=cache
        )
        vae = AutoencoderKL.from_pretrained(self.base, subfolder="vae", cache_dir=cache)
        unet = UNet2DConditionModel.from_pretrained(self.base, subfolder="unet", cache_dir=cache)

        # The photo's latent goes in beside the noise: widen the first convolution to 8
        # channels (the new ones start at zero), then add IC-Light's trained offsets.
        with torch.no_grad():
            old = unet.conv_in
            conv = torch.nn.Conv2d(8, old.out_channels, old.kernel_size, old.stride, old.padding)
            conv.weight.zero_()
            conv.weight[:, :4].copy_(old.weight)
            conv.bias = old.bias
            unet.conv_in = conv
        forward = unet.forward

        def hooked(sample: Any, timestep: Any, encoder_hidden_states: Any, **kwargs: Any) -> Any:
            extra = dict(kwargs.get("cross_attention_kwargs") or {})
            cond = extra.pop("concat_conds").to(sample)
            cond = torch.cat([cond] * (sample.shape[0] // cond.shape[0]), dim=0)
            kwargs["cross_attention_kwargs"] = extra
            return forward(
                torch.cat([sample, cond], dim=1), timestep, encoder_hidden_states, **kwargs
            )

        unet.forward = hooked
        offsets = load_file(hf_hub_download(self.repo, self.weights, cache_dir=cache))
        merged = {k: v + offsets[k] if k in offsets else v for k, v in unet.state_dict().items()}
        unet.load_state_dict(merged, strict=True)
        for part in (text_encoder, vae, unet):
            part.to(device=env.device, dtype=dtype)

        scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            steps_offset=1,
        )
        parts = {
            "vae": vae,
            "text_encoder": text_encoder,
            "tokenizer": tokenizer,
            "unet": unet,
            "scheduler": scheduler,
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
        }
        t2i = StableDiffusionPipeline(**parts)
        i2i = StableDiffusionImg2ImgPipeline(**parts)
        for pipe in (t2i, i2i):
            pipe.set_progress_bar_config(disable=True)
        return t2i, i2i, vae

    def run(self, image: Any, params: dict[str, Any], progress: Progress, env: Env) -> Any:
        import torch
        from PIL import Image

        t2i, i2i, vae = self.model(env, progress)
        h, w = image.shape[:2]
        scale = self.SIZE / max(h, w)
        gw, gh = (max(256, round(v * scale / 64) * 64) for v in (w, h))
        photo = cv2.resize(image, (gw, gh), interpolation=cv2.INTER_AREA)
        with torch.no_grad():
            pixels = torch.from_numpy(photo * 2 - 1).permute(2, 0, 1)[None]
            pixels = pixels.to(device=vae.device, dtype=vae.dtype)
            cond = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor

        direction = str(params.get("direction", "left"))
        prompt = ", ".join(p for p in (str(params.get("prompt") or ""), self.QUALITY) if p)
        steps = self.STEPS
        runs = steps if direction == "front" else max(1, int(steps * self.DENOISE))

        def on_step(_pipe: Any, step: int, _t: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
            progress(0.1 + 0.85 * (step + 1) / runs, "Relighting")
            return kwargs

        common: dict[str, Any] = {
            "prompt": prompt,
            "negative_prompt": self.NEGATIVE,
            "num_inference_steps": steps,
            "guidance_scale": 2.0,
            "generator": torch.Generator(device="cpu").manual_seed(int(params.get("seed", 0))),
            "cross_attention_kwargs": {"concat_conds": cond},
            "callback_on_step_end": on_step,
        }
        progress(0.1, "Relighting")
        gradient = _light_gradient(direction, (gh, gw))
        if gradient is None:
            result = t2i(width=gw, height=gh, **common).images[0]
        else:
            start = Image.fromarray((gradient * 255 + 0.5).astype(np.uint8), "RGB")
            result = i2i(image=start, strength=self.DENOISE, **common).images[0]
        out = (np.asarray(result.convert("RGB")) / 255.0).astype(np.float32)
        return _resize(out, (h, w))


def _light_gradient(direction: str, shape: tuple[int, int]) -> Array | None:
    """IC-Light's starting image for light from `direction`: bright on that side, dark on the
    other. None for light from the front, which starts from noise alone."""
    h, w = shape
    xs = np.linspace(0.0, 1.0, w, dtype=np.float32)
    ys = np.linspace(0.0, 1.0, h, dtype=np.float32)
    ramps = {
        "left": np.tile(1 - xs, (h, 1)),
        "right": np.tile(xs, (h, 1)),
        "top": np.tile((1 - ys)[:, None], (1, w)),
        "bottom": np.tile(ys[:, None], (1, w)),
    }
    if direction == "back":
        v, u = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot(u / w - 0.5, v / h - 0.5) * 2
        ramp = np.clip(r, 0, 1)
    elif direction in ramps:
        ramp = ramps[direction]
    else:
        return None
    return cast(Array, np.repeat((0.1 + 0.8 * ramp)[..., None], 3, axis=2))


def _resize(image: Array, shape: tuple[int, int]) -> Array:
    h, w = shape
    if image.shape[:2] == (h, w):
        return image
    shrinking = h < image.shape[0]
    flags = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cast(Array, np.clip(cv2.resize(image, (w, h), interpolation=flags), 0, 1))
