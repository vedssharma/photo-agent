"""Make small DNG files for tests: a known scene, "shot" by a made-up camera.

The camera sees linear Rec.2020 light through per-channel sensitivities, with an RGGB
Bayer filter; its DNG color matrix says exactly that, so a correct developer recovers the
scene (scaled to the camera's white level) from the mosaic.
"""

from __future__ import annotations

import io

import numpy as np
import numpy.typing as npt
import tifffile

XYZ_TO_REC2020 = np.array(
    [
        [1.7166512, -0.3556708, -0.2533663],
        [-0.6666844, 1.6164812, 0.0157685],
        [0.0176399, -0.0427706, 0.9421031],
    ]
)
SRGB_TO_REC2020 = np.array(
    [
        [0.6274039, 0.3292830, 0.0433131],
        [0.0690973, 0.9195404, 0.0113623],
        [0.0163914, 0.0880133, 0.8955953],
    ]
)
SENSITIVITY = np.array([0.5, 1.0, 0.7])
"""How strongly the made-up sensor responds to red, green, and blue (green is strongest,
like most cameras), which is what its as-shot white balance undoes."""
BLACK, WHITE = 512, 16383


def srgb_to_linear_rec2020(srgb: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """sRGB-encoded pixels as linear Rec.2020 light."""
    x = srgb.astype(np.float64)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    out: npt.NDArray[np.float64] = linear @ SRGB_TO_REC2020.T
    return out


def make_dng(
    linear_rec2020: npt.NDArray[np.float64],
    *,
    make: str = "Fixture",
    model: str = "RAW Camera",
    exposure: float = 1.0,
    lens: str = "",
    focal_length: float = 0.0,
    aperture: float = 0.0,
) -> bytes:
    """A DNG of the scene, `exposure` times as bright on the sensor; light above the
    sensor's white level clips, channel by channel, as it would in a real camera."""
    h, w = (s - s % 2 for s in linear_rec2020.shape[:2])
    cam = linear_rec2020[:h, :w] * SENSITIVITY * exposure
    mosaic = np.empty((h, w))
    mosaic[0::2, 0::2] = cam[0::2, 0::2, 0]
    mosaic[0::2, 1::2] = cam[0::2, 1::2, 1]
    mosaic[1::2, 0::2] = cam[1::2, 0::2, 1]
    mosaic[1::2, 1::2] = cam[1::2, 1::2, 2]
    raw = np.clip(mosaic * (WHITE - BLACK) + BLACK, 0, WHITE).astype(np.uint16)
    color_matrix = np.diag(SENSITIVITY) @ XYZ_TO_REC2020
    neutral = SENSITIVITY / SENSITIVITY.max()
    tags = [
        (50706, "B", 4, (1, 4, 0, 0), True),  # DNGVersion
        (50707, "B", 4, (1, 1, 0, 0), True),  # DNGBackwardVersion
        (50708, "s", 0, f"{make} {model}", True),  # UniqueCameraModel
        (271, "s", 0, make, True),
        (272, "s", 0, model, True),
        (306, "s", 0, "2026:10:12 09:30:00", True),  # DateTime
        (33421, "H", 2, (2, 2), True),  # CFARepeatPatternDim
        (33422, "B", 4, (0, 1, 1, 2), True),  # CFAPattern: RGGB
        (50710, "B", 3, (0, 1, 2), True),  # CFAPlaneColor
        (50714, "H", 1, (BLACK,), True),  # BlackLevel
        (50717, "H", 1, (WHITE,), True),  # WhiteLevel
        (50721, "2i", 9, _rationals(color_matrix.flatten(), signed=True), True),
        (50778, "H", 1, (21,), True),  # CalibrationIlluminant1: D65
        (50728, "2I", 3, _rationals(neutral, signed=False), True),  # AsShotNeutral
    ]
    # Shot details in the main IFD, as TIFF/EP has them.
    if lens:
        tags.append((42036, "s", 0, lens, True))  # LensModel
    if focal_length:
        tags.append((37386, "2I", 1, _rationals(np.array([focal_length]), signed=False), True))
    if aperture:
        tags.append((33437, "2I", 1, _rationals(np.array([aperture]), signed=False), True))
    buf = io.BytesIO()
    tifffile.imwrite(buf, raw, photometric=32803, subfiletype=0, extratags=tags, metadata=None)
    return buf.getvalue()


def _rationals(values: npt.NDArray[np.float64], *, signed: bool) -> list[int]:
    out: list[int] = []
    for v in values:
        out += [round(float(v) * 1_000_000), 1_000_000]
    return out
