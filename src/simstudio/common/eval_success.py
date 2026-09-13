"""Pick-and-place success check for the ``simple_pick`` MuJoCo scene."""

from __future__ import annotations

# Container body origin in SO101/scenes/simple_pick/scene.xml
CONTAINER_XY = (0.07, 0.065)
CONTAINER_HALF_XY = 0.04
CONTAINER_Z_MAX = 0.025


def check_pick_success(block_pos: tuple[float, float, float] | None) -> bool:
    """Return True when the cube rests inside the pick container."""
    if block_pos is None:
        return False
    x, y, z = block_pos
    cx, cy = CONTAINER_XY
    return abs(x - cx) <= CONTAINER_HALF_XY and abs(y - cy) <= CONTAINER_HALF_XY and z <= CONTAINER_Z_MAX


def cube_over_container(
    block_pos: tuple[float, float, float] | None,
    *,
    slack: float = 0.02,
) -> bool:
    """True when the cube XY is over the container (any height)."""
    if block_pos is None:
        return False
    x, y, _z = block_pos
    cx, cy = CONTAINER_XY
    half = CONTAINER_HALF_XY + slack
    return abs(x - cx) <= half and abs(y - cy) <= half


def gripper_near_cube(
    ee_pos: tuple[float, float, float],
    block_pos: tuple[float, float, float] | None,
    *,
    radius_m: float = 0.05,
    xy_m: float | None = None,
    z_min_m: float = -0.02,
    z_max_m: float = 0.08,
) -> bool:
    """True when a gripper-frame site is close enough to the cube to close.

    Prefer ``gripperframe`` (fingertips), not ``wrist_site`` (~10 cm proximal).
    Default is a 5 cm 3-D radius around the 3 cm cube. ``xy_m`` keeps the older
    XY+Z test for tests that still pass it.
    """
    if block_pos is None:
        return False
    dx = ee_pos[0] - block_pos[0]
    dy = ee_pos[1] - block_pos[1]
    dz = ee_pos[2] - block_pos[2]
    if xy_m is not None:
        return (dx * dx + dy * dy) ** 0.5 <= xy_m and z_min_m <= dz <= z_max_m
    return (dx * dx + dy * dy + dz * dz) ** 0.5 <= radius_m
