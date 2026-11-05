"""Encode metric depth for ACT (ResNet expects 3-channel images).

LeRobot ACT requires all ``observation.images.*`` to share the same shape, so
raw HxWx1 depth cannot sit next to RGB. We store a 3-channel gray rendering of
metric depth under ``camera_top_depth``; the values still come from the top
depth buffer (near=bright).
"""

from __future__ import annotations

import numpy as np


DEPTH_CAM_KEY = "camera_top_depth"


def depth_m_to_rgb3(
    depth_m: np.ndarray,
    *,
    near_m: float = 0.05,
    far_m: float = 2.0,
) -> np.ndarray:
    """Map metric depth (meters) to HxWx3 uint8 gray (near=bright)."""
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > near_m) & (depth < far_m)
    out = np.zeros((*depth.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return out
    lo = float(np.percentile(depth[valid], 5))
    hi = float(np.percentile(depth[valid], 95))
    hi = max(hi, lo + 1e-3)
    norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    gray = (255.0 * (1.0 - norm)).astype(np.uint8)
    gray[~valid] = 0
    out[..., 0] = gray
    out[..., 1] = gray
    out[..., 2] = gray
    return out
