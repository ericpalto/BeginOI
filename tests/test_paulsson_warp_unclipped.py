from __future__ import annotations

import numpy as np

from beginoi.plants.paulsson_machine_plant import _warp as plant_warp
from beginoi.tasks.single_paulsson.mismatch import _warp as mismatch_warp


def test_paulsson_warp_is_not_clipped_to_unit_box() -> None:
    U = np.array([[0.0, 0.0], [0.8, 0.1], [1.0, 1.0]], dtype=float)
    s = np.array([1.25, 0.75], dtype=float)
    t = np.array([0.12, -0.08], dtype=float)
    expected = U * s[None, :] + t[None, :]

    uw_plant = plant_warp(U, s=s, t=t)
    uw_mismatch = mismatch_warp(U, s=s, t=t)

    assert np.allclose(uw_plant, expected)
    assert np.allclose(uw_mismatch, expected)
    assert float(np.max(uw_plant)) > 1.0
    assert float(np.min(uw_plant)) < 0.0
