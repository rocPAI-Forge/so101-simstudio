"""Quota helpers for the Lab 01 yaw-aware collector (no MuJoCo)."""

from simstudio.common.lab01_spawn_quota import (
    XY_CELL_NAMES,
    next_deficit,
    split_quota,
    xy_cell,
)


def test_xy_cell_matches_named_ranges() -> None:
    assert xy_cell(0.27, 0.18) == "near_low_y"
    assert xy_cell(0.32, 0.18) == "far_low_y"
    assert xy_cell(0.27, 0.22) == "near_high_y"
    assert xy_cell(0.32, 0.22) == "far_high_y"
    assert xy_cell(0.30, 0.20) == "far_high_y"


def test_split_quota_puts_remainder_on_low_y() -> None:
    assert split_quota(100, 4, [0, 1, 2, 3]) == [25, 25, 25, 25]
    assert split_quota(50, 6, [0, 5, 1, 4, 2, 3]) == [9, 8, 8, 8, 8, 9]


def test_top_up_deficits_prefer_empty_low_y() -> None:
    """Existing 50-ep campaign: low-Y cells are the holes."""
    xy_quotas = split_quota(100, 4, [0, 1, 2, 3])
    xy_wins = [9, 8, 18, 15]  # near_low, far_low, near_high, far_high
    assert XY_CELL_NAMES[0] == "near_low_y"
    need = []
    cursor = 0
    for _ in range(50):
        idx = next_deficit(xy_wins, xy_quotas, cursor)
        assert idx is not None
        need.append(XY_CELL_NAMES[idx])
        xy_wins[idx] += 1
        cursor = idx + 1
    assert xy_wins == [25, 25, 25, 25]
    assert need.count("far_low_y") == 17
    assert need.count("near_low_y") == 16
    assert need.count("far_high_y") == 10
    assert need.count("near_high_y") == 7
