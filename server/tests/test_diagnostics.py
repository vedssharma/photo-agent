import numpy as np

from photo_agent import diagnostics


def test_measures_clipping_and_saturation() -> None:
    x = np.full((10, 10, 3), 0.5, np.float32)
    x[:2] = 1.0  # 20% white
    x[2:3] = 0.0  # 10% black
    x[3:4] = [1.0, 0.0, 0.0]  # 10% saturated red (also clipped)
    m = diagnostics.measure(x)
    assert m.clipped_highlights == 0.3
    assert m.crushed_shadows == 0.1
    assert m.oversaturated == 0.1


def test_warns_only_about_new_problems() -> None:
    calm = diagnostics.Measurements(
        clipped_highlights=0.0, crushed_shadows=0.0, oversaturated=0.0, mean_brightness=0.4
    )
    blown = calm.model_copy(update={"clipped_highlights": 0.2, "oversaturated": 0.1})
    found = diagnostics.warnings(calm, blown)
    assert len(found) == 2
    assert "blown-out" in found[0]
    # Already-blown sky in the original is not the agent's doing.
    assert diagnostics.warnings(blown, blown) == []
    assert "No overshoots" in diagnostics.report(calm, calm)
