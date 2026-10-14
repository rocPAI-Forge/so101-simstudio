"""Spawn-coverage quotas for the Lab 01 yaw-aware collector.

Kept free of MuJoCo / LeRobot so tests and top-up dry-runs can import it.
The 2×2 cells match the 8×7 cm eval randomization box.
"""

from __future__ import annotations

XY_CELLS: tuple[tuple[str, list[float], list[float]], ...] = (
    ("near_low_y", [0.26, 0.30], [0.165, 0.20]),
    ("far_low_y", [0.30, 0.34], [0.165, 0.20]),
    ("near_high_y", [0.26, 0.30], [0.20, 0.235]),
    ("far_high_y", [0.30, 0.34], [0.20, 0.235]),
)
XY_CELL_NAMES = [name for name, _, _ in XY_CELLS]
YAW_REMAINDER_ORDER = [0, 5, 1, 4, 2, 3]
XY_REMAINDER_ORDER = [0, 1, 2, 3]  # low-Y cells first


def xy_cell(x: float, y: float) -> str:
    return ("far" if x >= 0.30 else "near") + "_" + ("high_y" if y >= 0.20 else "low_y")


def split_quota(n: int, k: int, remainder_order: list[int]) -> list[int]:
    """Split n successes into k bins; leftover 1s follow remainder_order."""
    quotas = [n // k] * k
    for i in range(n % k):
        quotas[remainder_order[i]] += 1
    return quotas


def next_deficit(wins: list[int], quotas: list[int], cursor: int) -> int | None:
    """Round-robin index still below quota, or None if every bin is filled."""
    if all(wins[i] >= quotas[i] for i in range(len(quotas))):
        return None
    for step in range(len(quotas)):
        idx = (cursor + step) % len(quotas)
        if wins[idx] < quotas[idx]:
            return idx
    return None
