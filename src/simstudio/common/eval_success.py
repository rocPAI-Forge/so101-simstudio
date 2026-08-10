"""Pick-and-place success check for the ``simple_pick`` MuJoCo scene."""

from __future__ import annotations

# Container body origin in SO101/scenes/simple_pick/scene.xml
CONTAINER_XY = (0.3, 0.2)
CONTAINER_HALF_XY = 0.04
CONTAINER_Z_MAX = 0.025


def check_pick_success(block_pos: tuple[float, float, float] | None) -> bool:
    """Return True when the cube rests inside the pick container."""
    if block_pos is None:
        return False
    x, y, z = block_pos
    cx, cy = CONTAINER_XY
    return abs(x - cx) <= CONTAINER_HALF_XY and abs(y - cy) <= CONTAINER_HALF_XY and z <= CONTAINER_Z_MAX
