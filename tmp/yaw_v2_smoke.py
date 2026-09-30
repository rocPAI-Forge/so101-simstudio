#!/usr/bin/env python3
"""Six v2 episodes, no dataset write."""

import importlib.util
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

spec = importlib.util.spec_from_file_location("yaw", "labs/lab01_pnp/auto_collect_yaw.py")
yaw = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(yaw)
yaw.STEP_SLEEP_S = 0.0

robot = yaw.build_robot(headless=True)
robot.connect()
wins = 0
try:
    for i in range(6):
        yaw_bin = i % 6
        _cell, xr, yr = yaw.XY_CELLS[i % 4]
        robot.config.cube_random_yaw_range = [
            float(yaw.YAW_BIN_EDGES_DEG[yaw_bin]),
            float(yaw.YAW_BIN_EDGES_DEG[yaw_bin + 1]),
        ]
        robot.config.cube_random_x_range = list(xr)
        robot.config.cube_random_y_range = list(yr)
        ok = yaw.run_episode(
            robot,
            i,
            random_cube=True,
            start_jitter_rad=0.06,
            allow_regrasp=True,
        )
        wins += int(bool(ok))
        print(f"SMOKE i={i} success={bool(ok)} wins={wins}", flush=True)
finally:
    robot.disconnect()
print(f"SMOKE_DONE wins={wins}/6", flush=True)
