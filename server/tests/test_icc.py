"""ICC profiles in and out, and soft-proofing for sRGB and Display P3."""

import io

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient
from PIL import Image, ImageCms
from raw_helpers import make_dng

from photo_agent import color, icc, imaging

P3_RED = np.array([[[0.9, 0.12, 0.1]]], dtype=np.float32)
"""A red Display P3 can show and sRGB cannot (encoded P3 values)."""


@pytest.mark.parametrize("space", ["display-p3", "rec2020"])
def test_built_profiles_describe_their_space(space: color.Space) -> None:
    data = icc.profile(space)
    parsed = icc.parse(data)
    assert parsed is not None
    assert np.allclose(parsed.primaries(), color.PRIMARIES[space], atol=1e-3)
    assert not parsed.is_srgb() and not parsed.fits_in("srgb")
    # LittleCMS reads them the same way: colors that fit sRGB convert alike.
    lcms = ImageCms.ImageCmsProfile(io.BytesIO(data))
    assert ImageCms.getProfileDescription(lcms).strip() == color.SPACE_NAMES[space]
    rgb = np.array([[[150, 120, 100], [90, 140, 160], [128, 128, 128]]], np.uint8)
    theirs = ImageCms.profileToProfile(
        Image.fromarray(rgb), lcms, ImageCms.createProfile("sRGB"), outputMode="RGB"
    )
    assert theirs is not None
    ours = color.convert((rgb / 255).astype(np.float32), space, "srgb") * 255
    assert np.abs(np.asarray(theirs, np.float32) - ours).max() < 2


def test_srgb_profiles_are_recognized() -> None:
    parsed = icc.parse(icc.SRGB_ICC)
    assert parsed is not None and parsed.is_srgb()


def test_display_p3_photos_are_edited_in_the_wide_space() -> None:
    jpeg = imaging.encode_jpeg(
        np.repeat(np.repeat(P3_RED, 16, 0), 16, 1), quality=100, profile=icc.profile("display-p3")
    )
    img = imaging.decode(jpeg)
    assert img.space == "rec2020"
    back = color.convert(img.pixels, img.space, "display-p3")[8, 8]
    assert np.allclose(back, P3_RED[0, 0], atol=0.02)
    assert color.out_of_gamut(img.pixels, img.space, "srgb").all()


def test_srgb_tagged_and_untagged_photos_decode_alike() -> None:
    rng = np.random.default_rng(3)
    pixels = rng.random((20, 20, 3)).astype(np.float32)
    tagged = imaging.decode(imaging.encode_png(pixels))
    plain = Image.fromarray(imaging.to_uint8(pixels))
    buf = io.BytesIO()
    plain.save(buf, format="PNG")
    untagged = imaging.decode(buf.getvalue())
    assert tagged.space == untagged.space == "srgb"
    assert np.array_equal(tagged.pixels, untagged.pixels)


def test_sixteen_bit_tiffs_with_a_profile_convert_in_float() -> None:
    ramp = np.linspace(0, 1, 2000, dtype=np.float32)
    rgb = np.stack([ramp, ramp * 0.5, ramp * 0.25], axis=-1)[None]
    stored = np.round(rgb * 65535).astype(np.uint16)
    buf = io.BytesIO()
    profile = icc.profile("display-p3")
    tifffile.imwrite(
        buf, stored, photometric="rgb", extratags=[(34675, "B", len(profile), profile, True)]
    )
    img = imaging.decode(buf.getvalue())
    assert img.space == "rec2020"
    back = color.convert(img.pixels, "rec2020", "display-p3")
    assert np.abs(back - rgb).max() < 2e-3
    assert len(np.unique(img.pixels[0, :, 0])) > 1500  # Not rounded to 8 bits on the way.


def wide_doc(client: TestClient) -> str:
    scene = np.full((60, 120, 3), 0.18)
    # A green beyond sRGB, inside Display P3.
    p3_green = np.array([0.1, 0.6, 0.1], dtype=np.float32)
    scene[:, 60:] = color.convert_linear(p3_green, "display-p3", "rec2020")
    res = client.post("/api/documents", files={"file": ("green.dng", make_dng(scene))})
    assert res.status_code == 201, res.text
    assert res.json()["color_space"] == "rec2020"
    return str(res.json()["id"])


def _profile_name(data: bytes) -> str:
    embedded = Image.open(io.BytesIO(data)).info["icc_profile"]
    return ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(embedded))).strip()


def test_wide_photos_preview_in_display_p3(client: TestClient) -> None:
    doc_id = wide_doc(client)
    res = client.get(f"/api/documents/{doc_id}/preview")
    assert _profile_name(res.content) == "Display P3"
    srgb = client.get(f"/api/documents/{doc_id}/preview", params={"proof": "srgb"})
    assert "sRGB" in _profile_name(srgb.content)


def test_soft_proof_paints_colors_that_will_not_fit(client: TestClient) -> None:
    doc_id = wide_doc(client)
    url = f"/api/documents/{doc_id}/preview"
    warned = imaging.decode(
        client.get(url, params={"proof": "srgb", "gamut_warning": True}).content
    )
    left, right = warned.pixels[:, 10:50].mean((0, 1)), warned.pixels[:, 70:110].mean((0, 1))
    assert np.allclose(right, [1, 0, 1], atol=0.05)  # Out of sRGB: magenta.
    assert abs(left[0] - left[1]) < 0.03  # The gray card fits and is left alone.
    # The green fits Display P3, so proofing for it warns about nothing.
    p3 = client.get(url, params={"proof": "display-p3", "gamut_warning": True})
    shown = imaging.decode(p3.content)
    assert color.out_of_gamut(shown.pixels, shown.space, "display-p3").mean() < 0.01
    assert color.convert(shown.pixels, shown.space, "srgb")[:, 70:110, 1].mean() > 0.6

    gamut = client.get(f"/api/documents/{doc_id}/gamut").json()
    assert gamut["color_space"] == "rec2020"
    assert 0.4 < gamut["outside_srgb"] < 0.6
    assert gamut["outside_display_p3"] < 0.01


def test_exports_carry_the_chosen_profile(client: TestClient) -> None:
    doc_id = wide_doc(client)
    url = f"/api/documents/{doc_id}/export"
    srgb = client.post(url, json={"format": "png"}).content
    p3 = client.post(url, json={"format": "png", "color_space": "display-p3"}).content
    assert "sRGB" in _profile_name(srgb)
    assert _profile_name(p3) == "Display P3"
    # The P3 file keeps the green that the sRGB one had to map into range.
    p3_green = imaging.decode(p3)
    assert color.out_of_gamut(p3_green.pixels, p3_green.space, "srgb")[:, 70:110].mean() > 0.9


def test_srgb_photos_are_unchanged(client: TestClient, photos: object) -> None:
    res = client.post(
        "/api/documents",
        files={"file": ("p.png", imaging.encode_png(np.full((8, 8, 3), 0.4, np.float32)))},
    )
    doc = res.json()
    assert doc["color_space"] == "srgb"
    preview = client.get(f"/api/documents/{doc['id']}/preview").content
    assert "sRGB" in _profile_name(preview)
