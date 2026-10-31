"""Compiled open → place → close controller (tool-only)."""

from __future__ import annotations

import numpy as np

from .ik import _q_from_obs, _send, servo_to, set_gripper
from .kinematics import end_effector
from .motion import approach_xy, move_xy, slide
from .perceive import find_green_cube, find_lower_handle

GRASP_CLEARANCE_M = 0.012
PULL_DISTANCE_M = 0.075
PLACE_BEHIND_M = 0.055
PLACE_Z = 0.11
LIFT_Z = 0.14
GRASP_Z = 0.012
GRIP_OPEN = 0.85
GRIP_CLOSE = -0.08


def _open_gripper(ctx) -> None:
    ctx.step({"gripper_delta": 1.2}, duration_s=0.6, include_cameras=False)


def _close_gripper(ctx, duration_s: float = 1.2) -> None:
    ctx.step({"gripper_delta": -0.9}, duration_s=duration_s, include_cameras=False)


def open_drawer(ctx) -> dict | None:
    obs = ctx.observe(include_cameras=True)
    point, pull = find_lower_handle(obs)
    slide(ctx, pull, 0.04, squeeze=False)
    ctx.step({"wrist_flex_rate": -1.2}, duration_s=0.55, include_cameras=False)
    grasp_xy = end_effector(ctx.observe(include_cameras=False))[:2].copy()
    grasped = False
    for clearance in (GRASP_CLEARANCE_M, 0.005):
        approach_xy(ctx, point[:2] + pull * clearance)
        _close_gripper(ctx)
        grasp_xy = end_effector(ctx.observe(include_cameras=False))[:2].copy()
        slide(ctx, pull, 0.02, squeeze=True)
        try:
            moved, _ = find_lower_handle(ctx.observe(include_cameras=True))
        except RuntimeError:
            moved = None
        handle_moved = moved is not None and float(np.dot(moved[:2] - point[:2], pull)) >= 0.01
        pinch_moved = float(
            np.dot(end_effector(ctx.observe(include_cameras=False))[:2] - grasp_xy, pull)
        ) >= 0.01
        if handle_moved or (moved is None and pinch_moved):
            grasped = True
            break
        _open_gripper(ctx)
        slide(ctx, -pull, 0.03, squeeze=False)
    if not grasped:
        return None
    slide(ctx, pull, PULL_DISTANCE_M - 0.02, squeeze=True)
    _open_gripper(ctx)
    traveled = float(np.dot(end_effector(ctx.observe(include_cameras=False))[:2] - grasp_xy, pull))
    opened = point.copy()
    opened[:2] = point[:2] + pull * max(traveled, PULL_DISTANCE_M - 0.01)
    return {"point": opened, "pull": pull, "grasp_xy": grasp_xy}


def _send_pose(ctx, q: np.ndarray, *, steps: int = 60) -> None:
    for _ in range(steps):
        if _send(ctx, q, GRIP_OPEN):
            return


def _ready_for_pick(ctx) -> None:
    """Fold up, then face forward, without the joint-home sweep that shuts the tray.

    A direct move to the home joints from the pull pose drags the arm across the
    open drawer. Folding first, then yawing, reaches a pick seed with the tray
    still open.
    """
    q = _q_from_obs(ctx.observe(include_cameras=False))
    folded = np.array([float(q[0]), -1.2, 0.4, 1.2, 0.17], dtype=float)
    facing = np.array([0.0, -1.2, 0.4, 1.2, 0.17], dtype=float)
    _send_pose(ctx, folded)
    _send_pose(ctx, facing)


def pick_and_place(ctx, grasp: dict) -> bool:
    pull = grasp["pull"]
    _open_gripper(ctx)
    slide(ctx, pull, 0.12, squeeze=False)
    _ready_for_pick(ctx)
    set_gripper(ctx, GRIP_OPEN)
    try:
        cube = find_green_cube(ctx.observe(include_cameras=True))
    except RuntimeError:
        return False
    cx, cy = float(cube[0]), float(cube[1])
    # Heights match what the folded pick seed can reach; the old home seed sat lower.
    waypoints = [
        (np.array([0.24, 0.34, 0.08]), 0.02),
        (np.array([0.28, 0.32, 0.04]), 0.015),
        (np.array([cx, cy + 0.01, 0.025]), 0.012),
        (np.array([cx, cy - 0.005, GRASP_Z]), 0.01),
    ]
    for target, tol in waypoints:
        if not servo_to(ctx, target, gripper=GRIP_OPEN, steps=180, xy_tol=tol, z_tol=0.01):
            return False
    set_gripper(ctx, GRIP_CLOSE, steps=50)
    q_hold = _q_from_obs(ctx.observe(include_cameras=False))
    for _ in range(20):
        _send(ctx, q_hold, -0.05)
    grip = float(ctx.observe(include_cameras=False)["joints"]["gripper"]["position_rad"])
    if grip <= -0.05:
        return False
    hold = end_effector(ctx.observe(include_cameras=False)).copy()
    servo_to(
        ctx,
        np.array([hold[0], hold[1], LIFT_Z]),
        gripper=max(grip, -0.05),
        steps=160,
        lock_wrist=False,
        xy_tol=0.025,
        z_tol=0.025,
    )
    ee = end_effector(ctx.observe(include_cameras=False))
    try:
        lifted = find_green_cube(ctx.observe(include_cameras=True))
        held = float(lifted[2]) >= 0.035 and float(np.linalg.norm(lifted[:2] - ee[:2])) < 0.10
    except RuntimeError:
        held = float(ee[2]) >= 0.05
    if not held:
        return False
    try:
        handle, _ = find_lower_handle(ctx.observe(include_cameras=True))
        handle_xy = handle[:2]
    except RuntimeError:
        handle_xy = grasp["point"][:2]
    place_xy = handle_xy - pull * PLACE_BEHIND_M
    hold_grip = max(grip, -0.05)
    here = end_effector(ctx.observe(include_cameras=False))
    servo_to(
        ctx,
        np.array([here[0], here[1], 0.16]),
        gripper=hold_grip,
        steps=100,
        xy_tol=0.025,
        z_tol=0.02,
        lock_wrist=False,
    )
    servo_to(
        ctx,
        np.array([place_xy[0], place_xy[1], 0.16]),
        gripper=hold_grip,
        steps=200,
        xy_tol=0.02,
        z_tol=0.02,
        lock_wrist=False,
    )
    servo_to(
        ctx,
        np.array([place_xy[0], place_xy[1], PLACE_Z]),
        gripper=hold_grip,
        steps=120,
        xy_tol=0.02,
        z_tol=0.02,
        lock_wrist=False,
    )
    ee = end_effector(ctx.observe(include_cameras=False))
    if float(np.linalg.norm(ee[:2] - place_xy)) > 0.03:
        return False
    set_gripper(ctx, GRIP_OPEN, steps=30)
    for _ in range(15):
        _send(ctx, _q_from_obs(ctx.observe(include_cameras=False)), GRIP_OPEN)
    return True


def close_drawer(ctx, grasp: dict) -> bool:
    """Push the drawer shut on the pull axis recorded while opening.

    Velocity climbs after a place swing the wrist off the handle. This path
    pitches once on the robot side of the tray, then holds that wrist and
    servos onto the handle.
    """
    pull = np.asarray(grasp["pull"], dtype=float)
    open_xy = np.asarray(grasp["point"][:2], dtype=float)
    grasp_xy = np.asarray(grasp["grasp_xy"], dtype=float)
    try:
        set_gripper(ctx, GRIP_OPEN)
        here = end_effector(ctx.observe(include_cameras=False))
        servo_to(
            ctx,
            np.array([here[0], here[1], 0.16]),
            gripper=GRIP_OPEN,
            steps=80,
            xy_tol=0.03,
            z_tol=0.025,
            lock_wrist=False,
        )
        away = here[:2] + pull * 0.08
        servo_to(
            ctx,
            np.array([away[0], away[1], 0.16]),
            gripper=GRIP_OPEN,
            steps=100,
            xy_tol=0.03,
            z_tol=0.025,
            lock_wrist=False,
        )
        pitched = ctx.step({"wrist_flex_rate": -1.2}, duration_s=0.55, include_cameras=False)
        if pitched.get("terminated"):
            return bool(pitched.get("success"))
        wrist = _q_from_obs(ctx.observe(include_cameras=False))[3:]
        for clearance in (GRASP_CLEARANCE_M, 0.005):
            target = open_xy + pull * clearance
            servo_to(
                ctx,
                np.array([target[0], target[1], 0.062]),
                gripper=GRIP_OPEN,
                steps=140,
                xy_tol=0.012,
                z_tol=0.012,
                lock_wrist=True,
                wrist=wrist,
            )
            _close_gripper(ctx)
            before = end_effector(ctx.observe(include_cameras=False))[:2].copy()
            servo_to(
                ctx,
                np.array([grasp_xy[0], grasp_xy[1], 0.062]),
                gripper=GRIP_CLOSE,
                steps=160,
                xy_tol=0.008,
                z_tol=0.02,
                lock_wrist=True,
                wrist=wrist,
            )
            # Finish with the same squeeze the velocity close used.
            move_xy(ctx, grasp_xy, squeeze=True, tol=0.004)
            after = end_effector(ctx.observe(include_cameras=False))[:2]
            if float(np.dot(before - after, pull)) >= 0.02:
                break
            _open_gripper(ctx)
            slide(ctx, pull, 0.03, squeeze=False)
        result = ctx.step(
            {"vx": float(-pull[0] * 0.03), "vy": float(-pull[1] * 0.03), "gripper_delta": -0.05},
            duration_s=0.8,
            include_cameras=False,
        )
        if result.get("terminated"):
            return bool(result.get("success"))
    except RuntimeError as exc:
        if "terminated" in str(exc):
            return bool(ctx.check_success().get("success"))
        raise
    return bool(ctx.check_success().get("success"))


def main(ctx) -> bool:
    ctx.reset(include_cameras=False)
    grasp = open_drawer(ctx)
    if grasp is None:
        return bool(ctx.check_success().get("success"))
    pick_and_place(ctx, grasp)
    try:
        return close_drawer(ctx, grasp)
    except RuntimeError as exc:
        if "terminated" in str(exc):
            return bool(ctx.check_success().get("success"))
        raise
