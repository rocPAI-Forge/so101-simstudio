"""Open the drawer, place the green cube inside, then close it.

Drawer open/close use velocity control. Cube pick uses the Lab 01 style
position-mode Jacobian servo so both jaws can wrap the cube. Perception uses
only the overhead color and depth.

On some cabinet yaws the open tray collides with the arm home pose. Clearing
the handle (which can partially shove the drawer) then rehoming restores a
usable IK seed before the pick. Closing needs a long episode budget because
the place detour leaves the arm far from the handle.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

import mujoco as mj

from labs.rax_drawer.environment import RaxDrawerEnv
from labs.rax_drawer.open_close import (
    DrawerGrasp,
    _end_effector,
    _slide,
    close_drawer,
    open_drawer,
    perceive_lower_handle,
)

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
WRIST_GRASP = np.array([1.62, 0.17], dtype=float)
GRIPPER_OPEN = 0.85
GRIPPER_CLOSED = -0.08
GRASP_Z = 0.012
LIFT_Z = 0.14
PLACE_Z = 0.10
PLACE_BEHIND_HANDLE_M = 0.03


def perceive_cube(observation: dict) -> np.ndarray:
    """Return the green cube center in meters from the overhead camera."""
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
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    mask = (
        valid
        & (green > 100)
        & (green > red + 20)
        & (green > blue + 40)
        & (blue < 160)
        & (world[:, :, 2] > 0.004)
        & (world[:, :, 2] < 0.035)
    )
    labeled, count = ndimage.label(mask)
    best = None
    best_count = 0
    for index in range(1, count + 1):
        points = world[labeled == index]
        if len(points) < 40:
            continue
        span = points[:, :2].max(axis=0) - points[:, :2].min(axis=0)
        if float(span.max()) > 0.06:
            continue
        centroid = points.mean(axis=0)
        # Ignore green spill on the arm base / far behind the workspace.
        if float(centroid[1]) > 0.42 or float(centroid[1]) < 0.05:
            continue
        if len(points) > best_count:
            best_count = len(points)
            best = centroid
    if best is None:
        raise RuntimeError("overhead camera did not find the green cube")
    return best


def _q_arm(env: RaxDrawerEnv) -> np.ndarray:
    return np.array([float(env.data.ctrl[env.robot.act_ids[name]]) for name in ARM], dtype=float)


def _clip_arm(env: RaxDrawerEnv, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for index, name in enumerate(ARM):
        low, high = env.robot._mujoco_joint_range[name]
        out[index] = float(np.clip(out[index], low, high))
    return out


def _send_q(env: RaxDrawerEnv, q_arm: np.ndarray, gripper: float) -> None:
    if env.terminated:
        return
    action = {f"{name}.pos": float(q_arm[index]) for index, name in enumerate(ARM)}
    low, high = env.robot._mujoco_joint_range["gripper"]
    action["gripper.pos"] = float(np.clip(gripper, low, high))
    env.robot.send_action(action)
    env._update_score()
    if env.data.time - env.start_time >= env.task.episode_limit_s:
        env.terminated = True


def _servo_xyz(
    env: RaxDrawerEnv,
    target: np.ndarray,
    *,
    gripper: float,
    steps: int,
    xy_tol: float = 0.006,
    z_tol: float = 0.008,
    lock_wrist: bool = True,
) -> bool:
    cols = [env.robot.dof_ids[name] for name in (ARM[:3] if lock_wrist else ARM)]
    sid = env.robot.ee_site_id
    q = _q_arm(env)
    if lock_wrist:
        q[3:] = WRIST_GRASP
    for _ in range(steps):
        if env.terminated:
            return False
        mj.mj_forward(env.model, env.data)
        position = env.data.site_xpos[sid].copy()
        error = target - position
        if float(np.linalg.norm(error[:2])) <= xy_tol and abs(float(error[2])) <= z_tol:
            _send_q(env, q, gripper)
            return True
        command = 2.2 * error
        norm = float(np.linalg.norm(command))
        cap = min(0.012, 0.55 * float(np.linalg.norm(error)))
        if norm > cap > 0:
            command = command * (cap / norm)
        jacobian = np.zeros((3, env.model.nv))
        mj.mj_jacSite(env.model, env.data, jacobian, None, sid)
        j = jacobian[:, cols]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), command)
        full = np.zeros(5)
        full[: len(cols)] = dq
        trial = _clip_arm(env, q + np.clip(full, -0.55, 0.55) * 0.05)
        if lock_wrist:
            trial[3:] = WRIST_GRASP
        q = trial
        _send_q(env, q, gripper)
    return False


def _set_grip(env: RaxDrawerEnv, gripper: float, steps: int = 28) -> None:
    q = _q_arm(env)
    start = float(env.data.ctrl[env.robot.act_ids["gripper"]])
    for index in range(steps):
        if env.terminated:
            return
        alpha = (index + 1) / steps
        _send_q(env, q, (1.0 - alpha) * start + alpha * gripper)


def _velocity_to(
    env: RaxDrawerEnv,
    target: np.ndarray,
    *,
    timeout_s: float,
    tol: float = 0.015,
    squeeze: float = 0.0,
) -> bool:
    """World-frame velocity servo via env.step (bounded duration per call)."""
    elapsed = 0.0
    while elapsed < timeout_s and not env.terminated:
        delta = target - _end_effector(env)
        if float(np.linalg.norm(delta)) <= tol:
            return True
        norm = float(np.linalg.norm(delta)) + 1e-6
        speed = min(env.task.max_linear_speed_mps, max(norm / 0.25, 0.015))
        action = {
            "vx": float(delta[0] / norm * speed),
            "vy": float(delta[1] / norm * speed),
            "vz": float(delta[2] / norm * speed),
        }
        if squeeze:
            action["gripper_delta"] = float(squeeze)
        env.step(action, duration_s=0.2, include_cameras=False)
        elapsed += 0.2
    return float(np.linalg.norm(target - _end_effector(env))) <= tol * 1.5


def _clear_open_drawer(env: RaxDrawerEnv, grasp: DrawerGrasp) -> None:
    """Release the handle and slide toward the arm mount before rehoming.

    A direct rehome after the pull parks the pinch on the open tray for some
    cabinet yaws and jams the arm.
    """
    if not env.terminated:
        env.step({"gripper_delta": 1.2}, duration_s=0.8, include_cameras=False)
    _slide(env, grasp.pull, 0.10, squeeze=False)


def pick_and_place_cube(env: RaxDrawerEnv, grasp: DrawerGrasp) -> bool:
    """Pick the green cube and drop it into the open drawer tray."""
    if env.terminated:
        return False
    _clear_open_drawer(env, grasp)
    env.robot.reset_to_home_position()
    mj.mj_forward(env.model, env.data)
    try:
        cube = perceive_cube(env.observe(include_cameras=True))
    except RuntimeError:
        return False
    cx, cy = float(cube[0]), float(cube[1])
    q = _q_arm(env)
    q[3:] = WRIST_GRASP
    for _ in range(24):
        _send_q(env, q, GRIPPER_OPEN)
    waypoints = [
        (np.array([0.24, 0.34, 0.05]), 0.015),
        (np.array([0.28, 0.32, 0.03]), 0.01),
        (np.array([cx, cy + 0.01, 0.02]), 0.008),
        (np.array([cx, cy - 0.005, GRASP_Z]), 0.004),
    ]
    for target, tol in waypoints:
        _servo_xyz(
            env,
            target,
            gripper=GRIPPER_OPEN,
            steps=160,
            xy_tol=tol,
            z_tol=0.006,
            lock_wrist=True,
        )
    _set_grip(env, GRIPPER_CLOSED, steps=50)
    for _ in range(20):
        _send_q(env, _q_arm(env), -0.05)
    grip = float(env.data.qpos[env.robot.dof_ids["gripper"]])
    if grip <= -0.05:
        return False
    hold = _end_effector(env).copy()
    for height in np.linspace(hold[2], LIFT_Z, 45):
        _servo_xyz(
            env,
            np.array([hold[0], hold[1], height]),
            gripper=max(grip, -0.05),
            steps=3,
            lock_wrist=False,
            xy_tol=0.03,
            z_tol=0.012,
        )
    ee = _end_effector(env)
    try:
        lifted = perceive_cube(env.observe(include_cameras=True))
        held = float(lifted[2]) >= 0.035 and float(np.linalg.norm(lifted[:2] - ee[:2])) < 0.10
    except RuntimeError:
        # The cube is often hidden inside the jaws. Arm height is kinematics, not object pose.
        held = float(ee[2]) >= 0.05
    if not held:
        return False
    try:
        handle, _normal = perceive_lower_handle(env.observe(include_cameras=True))
        handle_xy = handle[:2]
    except RuntimeError:
        handle_xy = grasp.point[:2]
    # Drop just inside the opening. Driving the pinch onto the handle shuts the drawer.
    place_xy = handle_xy - grasp.pull * PLACE_BEHIND_HANDLE_M
    here = _end_effector(env)
    _velocity_to(env, np.array([here[0], here[1], 0.16]), timeout_s=3.0, tol=0.02, squeeze=-0.08)
    _velocity_to(
        env,
        np.array([place_xy[0], place_xy[1], 0.16]),
        timeout_s=6.0,
        tol=0.02,
        squeeze=-0.08,
    )
    _velocity_to(
        env,
        np.array([place_xy[0], place_xy[1], PLACE_Z]),
        timeout_s=3.0,
        tol=0.02,
        squeeze=-0.08,
    )
    ee = _end_effector(env)
    if float(np.linalg.norm(ee[:2] - place_xy)) > 0.03:
        return False
    _set_grip(env, GRIPPER_OPEN, steps=30)
    for _ in range(15):
        _send_q(env, _q_arm(env), GRIPPER_OPEN)
    return not env.terminated


def open_place_close(env: RaxDrawerEnv) -> bool:
    """Open the drawer, place the cube inside, and close the drawer."""
    grasp = open_drawer(env)
    if grasp is None or env.terminated:
        return env.check_success()
    pick_and_place_cube(env, grasp)
    if env.terminated:
        return env.check_success()
    return close_drawer(env, grasp)
