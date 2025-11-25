from __future__ import annotations

import numpy as np

from uav_acar_demo.ml.models import build_dense_baseline


def test_build_dense_baseline_forward() -> None:
    model = build_dense_baseline(input_dim=4)
    x = np.zeros((2, 4), dtype=np.float32)
    y = model.predict(x, verbose=0)
    assert y.shape == (2, 1)
