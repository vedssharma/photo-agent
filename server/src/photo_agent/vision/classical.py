"""Classical computer-vision fallbacks for the AI tasks: no model weights, CPU only.

They are rougher than the models (they guess from color, brightness, texture, and where
things usually are in a photo) but fast and always available, so every AI edit still does
something sensible without a GPU or downloaded weights. They also keep the test suite
independent of model downloads.

Every function takes RGB float32 pixels in 0..1 and returns a float32 mask in 0..1 at the
same size.
"""

from __future__ import annotations

from typing import Any, cast

import cv2
import numpy as np
import numpy.typing as npt

from photo_agent.imaging import Array, resize_long_edge

WORK_EDGE = 512
"""Long edge the fallbacks work at; masks are scaled back up and edge-refined."""

LUMA = np.array([0.2126, 0.7152, 0.0722], np.float32)

Box = tuple[float, float, float, float]
"""(left, top, right, bottom) in pixels."""


# Shared helpers


def gray(x: Array) -> Array:
    return cast(Array, np.clip(x @ LUMA, 0.0, 1.0).astype(np.float32))


def guided_filter(guide: Array, src: Array, radius: int, eps: float = 1e-3) -> Array:
    """Edge-aware smoothing of `src` that follows the edges of `guide` (He et al.), so a
    rough mask snaps to the photo's real edges (hair, branches, the horizon)."""
    size = (2 * radius + 1, 2 * radius + 1)

    def mean(a: npt.NDArray[Any]) -> Array:
        return cast(Array, cv2.boxFilter(a, -1, size, borderType=cv2.BORDER_REFLECT))

    mi, mp = mean(guide), mean(src)
    var = mean(guide * guide) - mi * mi
    cov = mean(guide * src) - mi * mp
    a = cov / (var + eps)
    b = mp - a * mi
    return cast(Array, np.clip(mean(a) * guide + mean(b), 0.0, 1.0).astype(np.float32))


def finish(mask: npt.NDArray[Any], image: Array, soften: float = 0.004) -> Array:
    """Scale a working-size mask up to the image and refine its edges against it."""
    h, w = image.shape[:2]
    m = np.asarray(mask, np.float32)
    if m.shape != (h, w):
        m = np.asarray(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR), np.float32)
    radius = max(2, round(max(h, w) * soften))
    return guided_filter(gray(image), m, radius)


def _work(image: Array) -> Array:
    return np.ascontiguousarray(resize_long_edge(image, WORK_EDGE))


def _hsv(x: Array) -> Array:
    return cast(Array, cv2.cvtColor(np.clip(x, 0.0, 1.0), cv2.COLOR_RGB2HSV))


def _keep_touching(mask: npt.NDArray[np.bool_], seeds: npt.NDArray[np.bool_]) -> npt.NDArray[Any]:
    """The connected parts of `mask` that overlap `seeds`."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return np.zeros_like(mask)
    keep = np.unique(labels[seeds & mask])
    keep = keep[keep != 0]
    return np.isin(labels, keep)


def _largest(mask: npt.NDArray[np.bool_], share: float = 0.0) -> npt.NDArray[Any]:
    """The largest connected part, and any others at least `share` of its size."""
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        return mask
    areas = stats[:, cv2.CC_STAT_AREA].copy()
    areas[0] = 0
    keep = areas >= max(1, areas.max() * share) if share > 0 else areas == areas.max()
    keep[0] = False
    return cast(npt.NDArray[Any], keep[labels])


def _grabcut(work: Array, init: npt.NDArray[np.uint8], iterations: int = 5) -> npt.NDArray[Any]:
    """Foreground from GrabCut seeded with a GC_* label map."""
    if not np.any((init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)):
        return np.zeros(init.shape, bool)
    img = (np.clip(work, 0, 1) * 255).astype(np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    labels = init.copy()
    try:
        cv2.grabCut(img, labels, (0, 0, 1, 1), bgd, fgd, iterations, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        # GrabCut needs both some background and some foreground; fall back to the seeds.
        labels = init
    return cast(npt.NDArray[Any], (labels == cv2.GC_FGD) | (labels == cv2.GC_PR_FGD))


def _rect_init(shape: tuple[int, int], box: Box) -> npt.NDArray[np.uint8]:
    h, w = shape
    init = np.full((h, w), cv2.GC_BGD, np.uint8)
    x0, y0 = max(0, int(box[0])), max(0, int(box[1]))
    x1, y1 = min(w, int(np.ceil(box[2]))), min(h, int(np.ceil(box[3])))
    init[y0:y1, x0:x1] = cv2.GC_PR_FGD
    return init


# Tasks


def sky(image: Array) -> Array:
    """Blue, or bright and flat, regions connected to the top of the frame."""
    work = _work(image)
    h = work.shape[0]
    hsv = _hsv(work)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lum = gray(work)
    gx = cv2.Sobel(cv2.GaussianBlur(lum, (0, 0), 1.0), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(cv2.GaussianBlur(lum, (0, 0), 1.0), cv2.CV_32F, 0, 1)
    texture = cv2.blur(np.sqrt(gx * gx + gy * gy), (7, 7))
    blue = (hue > 180) & (hue < 265) & (sat > 0.06) & (val > 0.15)
    overcast = (val > 0.62) & (sat < 0.22)
    candidate = (blue & (texture < 0.35)) | (overcast & (texture < 0.2))
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
    ).astype(bool)
    top = np.zeros_like(candidate)
    top[: max(1, h // 50)] = True
    found = _keep_touching(candidate, top)
    # Fill gaps such as birds, wires, and textured clouds inside the sky.
    closed = cv2.morphologyEx(found.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    return finish(closed.astype(np.float32), image)


def subject(image: Array) -> Array:
    """The main subject: GrabCut from a centered frame, kept to its largest part."""
    work = _work(image)
    h, w = work.shape[:2]
    mx, my = w * 0.06, h * 0.04
    init = _rect_init((h, w), (mx, my, w - mx, h))
    fg = _largest(_grabcut(work, init))
    return finish(fg.astype(np.float32), image)


def object_at(image: Array, box: list[float] | None, points: list[list[float]]) -> Array:
    """One object pointed at with a box (fractions) and/or points [x, y, include]."""
    work = _work(image)
    h, w = work.shape[:2]
    includes = [(p[0] * w, p[1] * h) for p in points if p[2]]
    excludes = [(p[0] * w, p[1] * h) for p in points if not p[2]]
    if box is not None:
        rect: Box = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
    elif includes:
        half = 0.2 * max(w, h)
        xs, ys = [p[0] for p in includes], [p[1] for p in includes]
        rect = (min(xs) - half, min(ys) - half, max(xs) + half, max(ys) + half)
    else:
        return np.zeros(image.shape[:2], np.float32)
    init = _rect_init((h, w), rect)
    dot = max(2, round(0.012 * max(w, h)))
    for x, y in includes:
        cv2.circle(init, (round(x), round(y)), dot, int(cv2.GC_FGD), -1)
    for x, y in excludes:
        cv2.circle(init, (round(x), round(y)), dot, int(cv2.GC_BGD), -1)
    fg = _grabcut(work, init)
    if includes:
        seeds = np.zeros_like(fg)
        for x, y in includes:
            seeds[min(h - 1, max(0, round(y))), min(w - 1, max(0, round(x)))] = True
        kept = _keep_touching(fg, seeds)
        fg = kept if kept.any() else fg
    else:
        # Parts of one thing can come apart (a head above a collar), so keep sizable ones.
        fg = _largest(fg, share=0.1)
    return finish(fg.astype(np.float32), image)


def skin_color(work: Array, strict: bool = False) -> npt.NDArray[Any]:
    """Pixels whose color is in the usual range of human skin tones. `strict` narrows the
    range to tell faces from similar colors such as blonde or light brown hair."""
    rgb = np.clip(work, 0, 1)
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb) * 255.0
    y, cr, cb = ycc[..., 0], ycc[..., 1], ycc[..., 2]
    top = rgb.max(axis=2)
    sat = (top - rgb.min(axis=2)) / np.maximum(top, 1e-3)
    # Lit skin is neither dark nor deeply saturated, which tells it apart from most hair.
    in_range = (cr > (141 if strict else 135)) & (cr < 180) & (cb > 80) & (cb < 135)
    return cast(npt.NDArray[Any], in_range & (y > 80) & (sat < 0.62))


def find_faces(image: Array, limit: int = 6) -> list[Box]:
    """Rough face boxes (pixels of `image`): compact, upright blobs of skin color."""
    work = _work(image)
    h, w = work.shape[:2]
    scale = image.shape[1] / w
    raw = skin_color(work, strict=True).astype(np.uint8)
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    skin = np.asarray(cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(skin, connectivity=8)
    faces: list[tuple[int, Box]] = []
    for i in range(1, count):
        x, y, bw, bh, area = (int(v) for v in stats[i])
        if area < 0.004 * h * w or bw < 8:
            continue
        # The face is the rows where the blob is widest; thinner rows above and below are
        # hair edges, ears, and the neck.
        widths = (labels[y : y + bh, x : x + bw] == i).sum(axis=1)
        rows = np.flatnonzero(widths >= 0.6 * widths.max())
        top, bottom = y + int(rows[0]), y + int(rows[-1]) + 1
        face_h = min(bottom - top, round(bw * 1.5))
        if not 0.7 <= face_h / bw <= 1.8 or area / (bw * bh) < 0.35:
            continue
        faces.append((area, (x * scale, top * scale, (x + bw) * scale, (top + face_h) * scale)))
    faces.sort(key=lambda f: -f[0])
    return [box for _, box in faces[:limit]]


def people(image: Array) -> Array:
    """Everyone in the photo: for each face found, the body below it, by GrabCut."""
    faces = find_faces(image)
    if not faces:
        return subject(image)
    work = _work(image)
    h, w = work.shape[:2]
    s = w / image.shape[1]
    init = np.full((h, w), cv2.GC_BGD, np.uint8)
    for x0, y0, x1, y1 in faces:
        fw, fh = (x1 - x0) * s, (y1 - y0) * s
        bx0, by0 = max(0, int(x0 * s - 1.3 * fw)), max(0, int(y0 * s - 0.5 * fh))
        bx1, by1 = min(w, int(x1 * s + 1.3 * fw)), min(h, int(y1 * s + 7 * fh))
        init[by0:by1, bx0:bx1] = cv2.GC_PR_FGD
    for x0, y0, x1, y1 in faces:
        cx, cy = (x0 + x1) / 2 * s, (y0 + y1) / 2 * s
        axes = (max(1, round((x1 - x0) * s * 0.3)), max(1, round((y1 - y0) * s * 0.3)))
        cv2.ellipse(init, (round(cx), round(cy)), axes, 0, 0, 360, int(cv2.GC_FGD), -1)
    return finish(_grabcut(work, init).astype(np.float32), image)


FACE_PARTS = ("skin", "face", "eyes", "lips", "teeth", "hair")


def face_parts(image: Array) -> dict[str, Array]:
    """Skin, face, eyes, lips, teeth, and hair, from where they sit in each face found."""
    h, w = image.shape[:2]
    parts = {name: np.zeros((h, w), np.float32) for name in FACE_PARTS}
    faces = find_faces(image)
    if not faces:
        return parts
    hsv = _hsv(image)
    sat, val = hsv[..., 1], hsv[..., 2]
    ycc = cv2.cvtColor(np.clip(image, 0, 1), cv2.COLOR_RGB2YCrCb)
    skin_px = skin_color(image, strict=True)

    def ellipse(cx: float, cy: float, ax: float, ay: float) -> npt.NDArray[Any]:
        canvas = np.zeros((h, w), np.uint8)
        axes = (max(1, round(ax)), max(1, round(ay)))
        cv2.ellipse(canvas, (round(cx), round(cy)), axes, 0, 0, 360, 1, -1)
        return canvas.astype(bool)

    for x0, y0, x1, y1 in faces:
        fw, fh = x1 - x0, y1 - y0
        cx = (x0 + x1) / 2
        face = ellipse(cx, y0 + fh * 0.5, fw * 0.5, fh * 0.58)
        eyes = ellipse(x0 + fw * 0.3, y0 + fh * 0.4, fw * 0.11, fh * 0.06) | ellipse(
            x0 + fw * 0.7, y0 + fh * 0.4, fw * 0.11, fh * 0.06
        )
        mouth = ellipse(cx, y0 + fh * 0.77, fw * 0.2, fh * 0.08)
        cr = ycc[..., 1]
        face_cr = float(np.median(cr[face & skin_px])) if (face & skin_px).any() else 0.55
        lips = mouth & (cr > face_cr + 0.02)
        teeth = mouth & (val > 0.55) & (sat < 0.3) & ~lips
        around = ellipse(cx, y0 + fh * 0.35, fw * 0.75, fh * 0.75)
        rows = np.arange(h)[:, None] < y0 + fh * 0.65
        hair = around & ~face & ~skin_px & rows & (val < 0.6)
        visible = ellipse(cx, y0 + fh * 0.75, fw * 0.85, fh * 1.3) & skin_px
        skin = (visible | (face & skin_px)) & ~eyes & ~mouth
        for name, found in (
            ("skin", skin),
            ("face", face),
            ("eyes", eyes),
            ("lips", lips),
            ("teeth", teeth),
            ("hair", hair),
        ):
            parts[name] = np.maximum(parts[name], found.astype(np.float32))
    return {name: finish(mask, image, soften=0.002) for name, mask in parts.items()}


def inpaint(image: Array, mask: npt.NDArray[Any], seed: int = 0) -> Array:
    """Fill the masked area from its surroundings.

    Fast-marching inpainting smears across large holes, so it runs at a scale where the
    hole is only a few pixels across, and the result is scaled back with grain matched to
    the area around the hole, so the fill does not look plastic.
    """
    hole = np.asarray(mask) > 0.5
    if not hole.any():
        return image
    h, w = hole.shape
    depth = float(cv2.distanceTransform(hole.astype(np.uint8), cv2.DIST_L2, 5).max())
    scale = min(1.0, 6.0 / max(depth, 1.0))
    sw, sh = max(8, round(w * scale)), max(8, round(h * scale))
    small = cv2.resize(image, (sw, sh), interpolation=cv2.INTER_AREA)
    small_hole = cv2.resize(hole.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
    small_hole = cv2.dilate(small_hole, np.ones((3, 3), np.uint8))
    img8 = (np.clip(small, 0, 1) * 255 + 0.5).astype(np.uint8)
    filled = cv2.inpaint(img8, small_hole, 3, cv2.INPAINT_TELEA).astype(np.float32) / 255
    up = np.asarray(cv2.resize(filled, (w, h), interpolation=cv2.INTER_CUBIC), np.float32)
    ring = cv2.dilate(hole.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~hole
    detail = image - cv2.GaussianBlur(image, (0, 0), 1.5)
    std = detail[ring].std(axis=0) if ring.any() else np.zeros(3, np.float32)
    noise = np.random.default_rng(seed).standard_normal((h, w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), 0.7) * std
    return cast(Array, np.where(hole[..., None], np.clip(up + noise, 0, 1), image))


# Generative fallbacks: without a generative model there is nothing to draw new content
# with, so these do the closest honest thing and the app says which backend ran.


def generate(image: Array, mask: npt.NDArray[Any] | None = None, seed: int = 0, **_: Any) -> Array:
    """Generative fill without a model: fill the area from its surroundings, as a removal
    would. The prompt cannot be followed."""
    if mask is None:
        return image
    return inpaint(image, mask, seed=seed)


def studio_backdrop(image: Array, mask: npt.NDArray[Any], seed: int = 0) -> Array:
    """A plain studio backdrop where `mask` is set: a soft gradient in a muted version of
    the old background's color, lighter behind the middle, with fine grain."""
    hole = np.asarray(mask) > 0.5
    if not hole.any():
        return image
    h, w = hole.shape
    base = image[hole].mean(axis=0)
    gray = float(base @ LUMA)
    color = 0.35 * base + 0.65 * gray
    color = color + (0.55 - float(color @ LUMA)) * 0.6
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xs / w - 0.5) / 0.8) ** 2 + ((ys / h - 0.4) / 0.9) ** 2)
    shade = (1.15 - 0.45 * np.clip(d, 0, 1.2))[..., None]
    noise = np.random.default_rng(seed).standard_normal((h, w, 1)).astype(np.float32)
    backdrop = np.clip(color * shade + 0.008 * noise, 0, 1).astype(np.float32)
    return cast(Array, np.where(hole[..., None], backdrop, image))
