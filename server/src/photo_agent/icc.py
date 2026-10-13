"""ICC profiles: read the ones photos come with, and write the ones exports carry.

Nearly every RGB profile in the wild (sRGB, Display P3 from phones and Macs, Adobe RGB and
ProPhoto from cameras and editors, Rec.2020) is a *matrix/TRC* profile: a tone curve per
channel and the three primaries' XYZ colors. Those are read here and applied in float, so
16-bit and wide-gamut photos convert without rounding to 8 bits or clipping to sRGB. Other
profiles (lookup-table ones, from printers and scanners) go through LittleCMS instead (see
`photo_agent.imaging`).

Profiles written here are ICC v2 matrix/TRC profiles with the sRGB tone curve, which every
browser, OS, and editor understands.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from PIL import ImageCms

from photo_agent import color

Array = npt.NDArray[np.float32]

D50 = np.array([0.9642, 1.0, 0.8249])
D65 = np.array([0.95047, 1.0, 1.08883])
BRADFORD = np.array(
    [[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]]
)


def adaptation(
    src_white: npt.NDArray[np.float64], dst_white: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Bradford chromatic adaptation from one white point to another."""
    cone_src = BRADFORD @ src_white
    cone_dst = BRADFORD @ dst_white
    out: npt.NDArray[np.float64] = np.linalg.inv(BRADFORD) @ np.diag(cone_dst / cone_src) @ BRADFORD
    return out


@dataclass
class MatrixProfile:
    """An RGB matrix/TRC profile, as linear RGB to XYZ (D65) and a tone curve per channel."""

    to_xyz: npt.NDArray[np.float64]
    """Linear RGB to XYZ with D65 white (the profile's D50 colorants adapted back)."""
    curves: tuple[Curve, Curve, Curve]

    def to_linear(self, encoded: Array) -> Array:
        """Encoded pixel values (0..1) as linear light, per the profile's tone curves."""
        out = np.empty_like(encoded, dtype=np.float32)
        for c, curve in enumerate(self.curves):
            out[..., c] = curve.apply(encoded[..., c])
        return out

    def primaries(self) -> list[tuple[float, float]]:
        """The (x, y) chromaticities of red, green, and blue."""
        cols = self.to_xyz.T
        return [(float(c[0] / c.sum()), float(c[1] / c.sum())) for c in cols]

    def is_srgb(self) -> bool:
        """Whether this is (to within rounding) the sRGB profile itself."""
        ref = np.array(color.PRIMARIES["srgb"])
        if np.abs(np.array(self.primaries()) - ref).max() > 0.003:
            return False
        probe = np.linspace(0, 1, 33, dtype=np.float32)
        srgb = color.decode_curve(probe)
        return all(np.abs(curve.apply(probe) - srgb).max() < 0.004 for curve in self.curves)

    def fits_in(self, space: color.Space) -> bool:
        """Whether every color this profile can hold is inside `space`."""
        tri = np.array(color.PRIMARIES[space])
        return all(_inside(np.array(p), tri, slack=0.004) for p in self.primaries())


def _inside(p: npt.NDArray[np.float64], tri: npt.NDArray[np.float64], slack: float) -> bool:
    a, b, c = tri
    v0, v1, v2 = c - a, b - a, p - a
    d00, d01, d02 = v0 @ v0, v0 @ v1, v0 @ v2
    d11, d12 = v1 @ v1, v1 @ v2
    inv = 1 / (d00 * d11 - d01 * d01)
    u = (d11 * d02 - d01 * d12) * inv
    v = (d00 * d12 - d01 * d02) * inv
    return bool(u >= -slack and v >= -slack and u + v <= 1 + slack)


@dataclass
class Curve:
    """A tone curve: a gamma, a sampled table, or an ICC parametric function."""

    gamma: float | None = None
    table: npt.NDArray[np.float64] | None = None
    params: tuple[int, tuple[float, ...]] | None = None

    def apply(self, x: Array) -> Array:
        x = np.clip(x, 0.0, 1.0)
        if self.table is not None:
            grid = np.linspace(0.0, 1.0, len(self.table))
            return np.asarray(np.interp(x, grid, self.table), dtype=np.float32)
        if self.params is not None:
            return np.asarray(_parametric(x, *self.params), dtype=np.float32)
        return np.asarray(np.power(x, self.gamma or 1.0), dtype=np.float32)


def _parametric(x: Array, kind: int, p: tuple[float, ...]) -> npt.NDArray[np.float64]:
    g = p[0]
    if kind == 0:
        return np.power(x, g)
    a, b = p[1], p[2]
    if kind == 1:
        return np.where(x >= -b / a, np.power(np.maximum(a * x + b, 0), g), 0.0)
    if kind == 2:
        c = p[3]
        return np.where(x >= -b / a, np.power(np.maximum(a * x + b, 0), g) + c, c)
    d = p[4]
    if kind == 3:
        c = p[3]
        return np.where(x >= d, np.power(np.maximum(a * x + b, 0), g), c * x)
    c, e, f = p[3], p[5], p[6]
    return np.where(x >= d, np.power(np.maximum(a * x + b, 0), g) + e, c * x + f)


PARAMETRIC_COUNTS = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}


def parse(data: bytes) -> MatrixProfile | None:
    """The matrix/TRC description of an RGB profile, or None for any other kind (or a
    damaged one)."""
    try:
        if len(data) < 132 or data[16:20] != b"RGB " or data[20:24] != b"XYZ ":
            return None
        (count,) = struct.unpack(">I", data[128:132])
        tags: dict[bytes, bytes] = {}
        for i in range(min(count, 100)):
            sig, offset, size = struct.unpack(">4sII", data[132 + 12 * i : 144 + 12 * i])
            tags[sig] = data[offset : offset + size]
        colorants = [_xyz(tags[name]) for name in (b"rXYZ", b"gXYZ", b"bXYZ")]
        curves = (_curve(tags[b"rTRC"]), _curve(tags[b"gTRC"]), _curve(tags[b"bTRC"]))
    except (KeyError, struct.error, ValueError, IndexError):
        return None
    to_xyz_d50 = np.stack(colorants, axis=1)
    # Colorants are adapted to the D50 connection space; bring them back to D65.
    return MatrixProfile(to_xyz=adaptation(D50, D65) @ to_xyz_d50, curves=curves)


def _s15(raw: bytes) -> float:
    return float(struct.unpack(">i", raw)[0]) / 65536.0


def _xyz(tag: bytes) -> npt.NDArray[np.float64]:
    if tag[:4] != b"XYZ ":
        raise ValueError("not an XYZ tag")
    return np.array([_s15(tag[8 + 4 * i : 12 + 4 * i]) for i in range(3)])


def _curve(tag: bytes) -> Curve:
    kind = tag[:4]
    if kind == b"curv":
        (n,) = struct.unpack(">I", tag[8:12])
        if n == 0:
            return Curve(gamma=1.0)
        if n == 1:
            return Curve(gamma=struct.unpack(">H", tag[12:14])[0] / 256.0)
        values = np.frombuffer(tag[12 : 12 + 2 * n], dtype=">u2").astype(np.float64)
        return Curve(table=values / 65535.0)
    if kind == b"para":
        (function,) = struct.unpack(">H", tag[8:10])
        n = PARAMETRIC_COUNTS[function]
        params = tuple(_s15(tag[12 + 4 * i : 16 + 4 * i]) for i in range(n))
        return Curve(params=(function, params))
    raise ValueError("unknown curve type")


# Writing profiles


def _s15_bytes(v: float) -> bytes:
    return struct.pack(">i", round(v * 65536))


def _xyz_tag(xyz: npt.NDArray[np.float64]) -> bytes:
    return b"XYZ \0\0\0\0" + b"".join(_s15_bytes(float(v)) for v in xyz)


def _text_tag(text: str) -> bytes:
    return b"text\0\0\0\0" + text.encode("ascii") + b"\0"


def _desc_tag(text: str) -> bytes:
    ascii_ = text.encode("ascii") + b"\0"
    return (
        b"desc\0\0\0\0"
        + struct.pack(">I", len(ascii_))
        + ascii_
        + struct.pack(">II", 0, 0)  # No Unicode description.
        + struct.pack(">HB", 0, 0)
        + bytes(67)  # No ScriptCode description.
    )


def _srgb_curve_tag(samples: int = 1024) -> bytes:
    x = np.linspace(0, 1, samples, dtype=np.float32)
    table = np.round(color.decode_curve(x) * 65535).astype(">u2")
    return b"curv\0\0\0\0" + struct.pack(">I", samples) + table.tobytes()


def _pad(b: bytes) -> bytes:
    return b + bytes(-len(b) % 4)


def build(space: color.Space) -> bytes:
    """An ICC v2 profile for `space` (its primaries, D65 white, and the sRGB tone curve)."""
    if space == "srgb":
        return SRGB_ICC
    to_xyz_d50 = adaptation(D65, D50) @ color.to_xyz_matrix(space)
    curve = _srgb_curve_tag()
    name = color.SPACE_NAMES[space]
    tags: list[tuple[bytes, bytes]] = [
        (b"desc", _desc_tag(name)),
        (b"cprt", _text_tag("No copyright, use freely")),
        (b"wtpt", _xyz_tag(D50)),
        (b"rXYZ", _xyz_tag(to_xyz_d50[:, 0])),
        (b"gXYZ", _xyz_tag(to_xyz_d50[:, 1])),
        (b"bXYZ", _xyz_tag(to_xyz_d50[:, 2])),
        (b"rTRC", curve),
        (b"gTRC", curve),
        (b"bTRC", curve),
    ]
    table_size = 4 + 12 * len(tags)
    offset = 128 + table_size
    table = struct.pack(">I", len(tags))
    body = b""
    shared: dict[bytes, int] = {}
    for sig, payload in tags:
        if payload in shared:  # The three tone curves share one copy.
            table += struct.pack(">4sII", sig, shared[payload], len(payload))
            continue
        shared[payload] = offset + len(body)
        table += struct.pack(">4sII", sig, offset + len(body), len(payload))
        body += _pad(payload)
    size = 128 + len(table) + len(body)
    header = (
        struct.pack(">I", size)
        + bytes(4)  # Preferred CMM: any.
        + bytes([2, 0x40, 0, 0])  # Version 2.4.
        + b"mntrRGB XYZ "
        + struct.pack(">6H", 2026, 1, 1, 0, 0, 0)
        + b"acsp"
        + b"APPL"
        + struct.pack(">I", 0)  # Flags.
        + b"\0\0\0\0\0\0\0\0"  # Device manufacturer and model.
        + bytes(8)  # Attributes.
        + struct.pack(">I", 0)  # Rendering intent: perceptual.
        + b"".join(_s15_bytes(float(v)) for v in D50)
        + b"phag"  # Creator.
    )
    header += bytes(128 - len(header))
    return header + table + body


SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
"""LittleCMS's own sRGB profile."""

PROFILES: dict[color.Space, bytes] = {}


def profile(space: color.Space) -> bytes:
    """The profile to embed in a file whose pixels are in `space` (built once)."""
    if space not in PROFILES:
        PROFILES[space] = build(space)
    return PROFILES[space]
