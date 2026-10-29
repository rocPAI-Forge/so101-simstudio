"""Open and close the lower drawer from the overhead camera and arm kinematics.

The handle position comes from the top color image and its depth. The arm mount
is fixed, so the pull direction is the horizontal normal of the handle bar that
points toward that mount. Drawer joint position is not read.
"""

from __future__ import annotations

import numpy as np

from labs.rax_drawer.environment import RaxDrawerEnv

ARM_MOUNT_XY = np.array([0.18, 0.48])
PULL_DISTANCE_M = 0.075
# Stop on the robot side of the perceived bar. Driving the pinch onto the
# centroid puts the upper jaw into the shelf above the drawer.
GRASP_CLEARANCE_M = 0.012


def perceive_lower_handle(observation: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower-handle point in meters and a unit pull direction in XY."""
    rgb = observation["cameras"]["top"]
    depth_m = observation["depth_images"]["top"].astype(np.float64) / 1000.0
    calibration = observation["calibration"]["top"]
    rotation = np.asarray(calibration["rotation_camera_to_world"], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(calibration["position_m"], dtype=np.float64)
    fx, fy = calibration["fx"], calibration["fy"]
    cx, cy = calibration["cx"], calibration["cy"]
    height, width = depth_m.shape
    rows, cols = np.mgrid[0:height, 0:width]
    valid = (depth_m > 0.2) & (depth_m < 2.0)
    camera_points = np.stack(
        [
            (cols - cx) / fx * depth_m,
            (cy - rows) / fy * depth_m,
            -depth_m,
        ],
        axis=-1,
    )
    world = camera_points @ rotation.T + origin
    handle = (
        valid
        & (rgb.max(axis=-1) < 50)
        & (world[:, :, 2] > 0.045)
        & (world[:, :, 2] < 0.095)
    )
    if int(handle.sum()) < 30:
        raise RuntimeError("overhead camera did not find the lower handle")
    points = world[handle]
    centroid = points.mean(axis=0)
    flat = points[:, :2] - centroid[:2]
    _, _, axes = np.linalg.svd(flat, full_matrices=False)
    normal = np.array([-axes[0, 1], axes[0, 0]])
    if float(np.dot(normal, ARM_MOUNT_XY - centroid[:2])) < 0:
        normal = -normal
    normal = normal / np.linalg.norm(normal)
    return centroid, normal


def _end_effector(env: RaxDrawerEnv) -> np.ndarray:
    """Arm end-effector position from the robot kinematics."""
    return env.data.site_xpos[env.robot.ee_site_id].copy()


def _slide(env: RaxDrawerEnv, direction: np.ndarray, distance_m: float, *, squeeze: bool) -> None:
    """Move the pinch horizontally. Direction is a unit vector in XY."""
    start = _end_effector(env)[:2].copy()
    speed = env.task.max_linear_speed_mps
    elapsed = 0.0
    limit = distance_m / speed * 2.0 + 0.4
    action = {"vx": float(direction[0] * speed), "vy": float(direction[1] * speed)}
    if squeeze:
        action["gripper_delta"] = -0.05
    while elapsed < limit and not env.terminated:
        traveled = float(np.dot(_end_effector(env)[:2] - start, direction))
        if traveled >= distance_m:
            return
        env.step(action, duration_s=0.2, include_cameras=False)
        elapsed += 0.2


def _approach(env: RaxDrawerEnv, target_xy: np.ndarray) -> None:
    elapsed = 0.0
    while elapsed < 6.0 and not env.terminated:
        delta = target_xy - _end_effector(env)[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.008:
            return
        speed = min(env.task.max_linear_speed_mps, max(distance / 0.2, 0.015))
        direction = delta / distance
        env.step(
            {"vx": float(direction[0] * speed), "vy": float(direction[1] * speed)},
            duration_s=0.2,
            include_cameras=False,
        )
        elapsed += 0.2


def open_and_close(env: RaxDrawerEnv) -> bool:
    """Grasp the perceived lower handle, pull it open, and return it."""
    point, pull = perceive_lower_handle(env.observe(include_cameras=True))
    # Pitching in place swings the gripper into a cabinet that sits close to the arm.
    _slide(env, pull, 0.04, squeeze=False)
    if not env.terminated:
        env.step({"wrist_flex_rate": -1.2}, duration_s=0.55, include_cameras=False)
    grasped = False
    for clearance in (GRASP_CLEARANCE_M, 0.005):
        if env.terminated:
            break
        _approach(env, point[:2] + pull * clearance)
        if env.terminated:
            break
        env.step({"gripper_delta": -0.9}, duration_s=1.2, include_cameras=False)
        grasp_xy = _end_effector(env)[:2].copy()
        _slide(env, pull, 0.02, squeeze=True)
        try:
            moved, _ = perceive_lower_handle(env.observe(include_cameras=True))
        except RuntimeError:
            moved = None
        handle_moved = moved is not None and float(np.dot(moved[:2] - point[:2], pull)) >= 0.01
        pinch_moved = float(np.dot(_end_effector(env)[:2] - grasp_xy, pull)) >= 0.01
        if handle_moved or (moved is None and pinch_moved):
            grasped = True
            break
        if not env.terminated:
            env.step({"gripper_delta": 1.2}, duration_s=0.6, include_cameras=False)
        _slide(env, -pull, 0.03, squeeze=False)
    if not grasped or env.terminated:
        return env.check_success()
    _slide(env, pull, PULL_DISTANCE_M - 0.02, squeeze=True)
    if env.terminated:
        return True
    elapsed = 0.0
    while elapsed < 8.0 and not env.terminated:
        delta = grasp_xy - _end_effector(env)[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= 0.004:
            break
        speed = min(0.04, max(distance / 0.2, 0.015))
        direction = delta / max(distance, 1e-6)
        env.step(
            {
                "vx": float(direction[0] * speed),
                "vy": float(direction[1] * speed),
                "gripper_delta": -0.05,
            },
            duration_s=0.2,
            include_cameras=False,
        )
        elapsed += 0.2
    if not env.terminated:
        env.step(
            {"vx": float(-pull[0] * 0.02), "vy": float(-pull[1] * 0.02), "gripper_delta": -0.05},
            duration_s=0.5,
            include_cameras=False,
        )
    return env.check_success()
