# /// script
# requires-python = ">=3.12"
# dependencies = ["scikit-image==0.26.0", "pillow==12.*", "pillow-heif==1.*", "numpy==2.*"]
# ///
"""Regenerate the test photo fixtures in fixtures/photos/.

Sources are public-domain or CC0 photos bundled with scikit-image (see
fixtures/README.md for credits). Run from the repo root:

    uv run fixtures/generate.py

Encoders differ across library versions, so re-running may change bytes but
not what each fixture is for. The committed files are the source of truth.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pillow_heif
import skimage.data
from PIL import Image

pillow_heif.register_heif_opener()

OUT = Path(__file__).parent / "photos"

EXIF_MAKE = 0x010F
EXIF_MODEL = 0x0110
EXIF_DATETIME = 0x0132
EXIF_ORIENTATION = 0x0112
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
EXIF_EXPOSURE_TIME = 0x829A
EXIF_ISO = 0x8827


def make_exif(
    *,
    make: str,
    model: str,
    taken: str,
    orientation: int = 1,
    iso: int | None = None,
    exposure: float | None = None,
    gps: tuple[float, float] | None = None,
) -> Image.Exif:
    exif = Image.Exif()
    exif[EXIF_MAKE] = make
    exif[EXIF_MODEL] = model
    exif[EXIF_DATETIME] = taken
    exif[EXIF_ORIENTATION] = orientation
    sub = exif.get_ifd(EXIF_IFD)
    if iso is not None:
        sub[EXIF_ISO] = iso
    if exposure is not None:
        sub[EXIF_EXPOSURE_TIME] = exposure
    if gps is not None:
        lat, lon = gps
        g = exif.get_ifd(GPS_IFD)
        g[1] = "N" if lat >= 0 else "S"
        g[2] = _dms(abs(lat))
        g[3] = "E" if lon >= 0 else "W"
        g[4] = _dms(abs(lon))
    return exif


def _dms(deg: float) -> tuple[float, float, float]:
    d = int(deg)
    m = int((deg - d) * 60)
    s = round((deg - d - m / 60) * 3600, 2)
    return (float(d), float(m), s)


def low_light(rgb: np.ndarray, seed: int) -> np.ndarray:
    """Simulate an underexposed, noisy, slightly cool high-ISO shot."""
    rng = np.random.default_rng(seed)
    linear = (rgb.astype(np.float32) / 255.0) ** 2.2
    linear *= 2.0**-3  # three stops under
    linear *= np.array([0.92, 0.97, 1.08], dtype=np.float32)  # cool cast
    noisy = linear + rng.normal(0.0, 0.004, linear.shape).astype(np.float32)
    out = np.clip(noisy, 0.0, 1.0) ** (1 / 2.2)
    return (out * 255.0).round().astype(np.uint8)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def record(name: str, img: Image.Image, **meta: Any) -> None:
        manifest.append(
            {"file": name, "width": img.width, "height": img.height, **meta}
        )

    # Portrait: a person, head and shoulders. JPEG with camera EXIF.
    astronaut = Image.fromarray(skimage.data.astronaut())
    astronaut.save(
        OUT / "portrait.jpg",
        quality=90,
        exif=make_exif(make="NASA", model="Fixture", taken="2026:10:08 10:00:00"),
    )
    record(
        "portrait.jpg",
        astronaut,
        format="JPEG",
        tags=["portrait", "person", "face"],
        source="astronaut",
    )

    # Same portrait stored sideways with EXIF Orientation 6: decoders must rotate
    # it 90 degrees clockwise to display it upright (like most phone photos).
    sideways = astronaut.transpose(Image.Transpose.ROTATE_90)
    sideways.save(
        OUT / "portrait-exif-rotated.jpg",
        quality=90,
        exif=make_exif(
            make="NASA", model="Fixture", taken="2026:10:08 10:00:00", orientation=6
        ),
    )
    record(
        "portrait-exif-rotated.jpg",
        astronaut,
        format="JPEG",
        orientation=6,
        stored_width=sideways.width,
        stored_height=sideways.height,
        tags=["portrait", "person", "face", "exif-orientation"],
        source="astronaut",
    )

    # Landscape: outdoor scene with a large sky, wider than tall. Lossless PNG.
    rocket = Image.fromarray(skimage.data.rocket())
    rocket.save(OUT / "landscape.png", optimize=True)
    record(
        "landscape.png",
        rocket,
        format="PNG",
        tags=["landscape", "outdoor", "sky"],
        source="rocket",
    )

    # Low light: an indoor scene pushed three stops under with sensor noise.
    coffee = skimage.data.coffee()
    dark = Image.fromarray(low_light(coffee, seed=8))
    dark.save(
        OUT / "low-light.jpg",
        quality=88,
        exif=make_exif(
            make="Fixture",
            model="Low Light",
            taken="2026:10:08 21:30:00",
            iso=6400,
            exposure=1 / 15,
        ),
    )
    record(
        "low-light.jpg",
        dark,
        format="JPEG",
        tags=["low-light", "underexposed", "noisy", "indoor"],
        source="coffee",
    )

    # Phone photo: HEIC like an iPhone produces, including GPS that export
    # should strip by default.
    cat = Image.fromarray(skimage.data.chelsea())
    cat.save(
        OUT / "phone.heic",
        quality=85,
        exif=make_exif(
            make="Apple",
            model="iPhone (fixture)",
            taken="2026:10:08 14:15:00",
            iso=64,
            exposure=1 / 120,
            gps=(37.7749, -122.4194),
        ),
    )
    record(
        "phone.heic",
        cat,
        format="HEIF",
        has_gps=True,
        tags=["heic", "phone", "pet", "gps"],
        source="chelsea",
    )

    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for entry in manifest:
        size = (OUT / entry["file"]).stat().st_size
        print(
            f"{entry['file']:<28} {entry['width']}x{entry['height']}  {size / 1024:.0f} KB"
        )


if __name__ == "__main__":
    main()
