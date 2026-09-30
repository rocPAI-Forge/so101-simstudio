#!/usr/bin/env python3
"""Lift-rate probe for the current yaw expert. Does not write a dataset."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "auto_collect_yaw", ROOT / "labs/lab01_pnp/auto_collect_yaw.py"
)
yaw = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["auto_collect_yaw"] = yaw
SPEC.loader.exec_module(yaw)

yaw.STEP_SLEEP_S = 0.0

N = 10
CONDITIONS: list[tuple[str, dict]] = [
    ("nominal", {}),
    ("dx+5mm", {"close_dx": 0.005}),
    ("dx-5mm", {"close_dx": -0.005}),
    ("dy+5mm", {"close_dy": 0.005}),
    ("dy-5mm", {"close_dy": -0.005}),
    ("roll+5deg", {"close_droll": float(np.deg2rad(5))}),
    ("roll-5deg", {"close_droll": float(np.deg2rad(-5))}),
    ("grip-0.17", {"gripper_closed": -0.17}),
    ("grip-0.17_dx+5mm", {"gripper_closed": -0.17, "close_dx": 0.005}),
    ("grip-0.17_dx-5mm", {"gripper_closed": -0.17, "close_dx": -0.005}),
]


def main() -> int:
    np.random.seed(20260927)
    robot = yaw.build_robot(headless=True)
    robot.connect()
    totals: list[tuple[str, int, int]] = []
    try:
        for name, kwargs in CONDITIONS:
            lifts = 0
            for trial in range(N):
                yaw_bin = trial % 6
                xy_idx = trial % 4
                _cell, x_range, y_range = yaw.XY_CELLS[xy_idx]
                robot.config.cube_random_yaw_range = [
                    float(yaw.YAW_BIN_EDGES_DEG[yaw_bin]),
                    float(yaw.YAW_BIN_EDGES_DEG[yaw_bin + 1]),
                ]
                robot.config.cube_random_x_range = list(x_range)
                robot.config.cube_random_y_range = list(y_range)
                lifted = yaw.run_episode(
                    robot,
                    trial,
                    random_cube=True,
                    stop_after_lift=True,
                    **kwargs,
                )
                lifts += int(bool(lifted))
                print(
                    f"PROBE condition={name} trial={trial + 1}/{N} lifted={int(bool(lifted))} "
                    f"running={lifts}/{trial + 1}",
                    flush=True,
                )
            totals.append((name, lifts, N))
            print(f"PROBE_SUM {name} {lifts}/{N}", flush=True)
    finally:
        robot.disconnect()
    print("PROBE_DONE", flush=True)
    for name, lifts, n in totals:
        print(f"  {name:22s} {lifts}/{n}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
