# Test photo fixtures

A small, license-clean set of photos covering the formats and situations the app must handle. `photos/manifest.json` describes each file, and `server/tests/test_fixtures.py` checks that every one decodes and matches it.

| File | Format | What it covers | Source and license |
| --- | --- | --- | --- |
| `portrait.jpg` | JPEG | Head-and-shoulders portrait of a person, camera EXIF | Eileen Collins, NASA; public domain |
| `portrait-exif-rotated.jpg` | JPEG | Same portrait stored sideways with EXIF Orientation 6, like most phone photos; must be rotated on load | As above |
| `landscape.png` | PNG | Wide outdoor scene at dusk with a big sky and point lights | DSCOVR launch, SpaceX; public domain |
| `low-light.jpg` | JPEG | Indoor scene three stops underexposed with high-ISO noise and a cool cast (simulated from the original) | Coffee cup, Rachel Michetti; CC0 |
| `phone.heic` | HEIC | iPhone-style HEIC with GPS in EXIF, which export should strip by default | Chelsea the cat, Stefan van der Walt; CC0 |

All sources are the sample images bundled with [scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html), which documents their licenses.

## Regenerating

```sh
uv run fixtures/generate.py
```

The script declares its own dependencies, so nothing extra needs installing. Encoder output varies across library versions; the committed files are the source of truth, so only regenerate when adding or changing a fixture.
