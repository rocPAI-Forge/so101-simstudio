"""Bounded motions through Session tools only."""

from __future__ import annotations

import numpy as np

from .kinematics import end_effector

SPEED = 0.04


def approach_xy(ctx, target_xy: np.ndarray, *, timeout_s: float = 6.0, tol: float = 0.008) -> bool:
    elapsed = 0.0
    while elapsed < timeout_s:
        obs = ctx.observe(include_cameras=False)
        if obs.get("sim_time_s", 0) >= 175:
            return False
        delta = np.asarray(target_xy, dtype=float) - end_effector(obs)[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= tol:
            return True
        speed = min(SPEED, max(distance / 0.2, 0.015))
        direction = delta / distance
        result = ctx.step(
            {"vx": float(direction[0] * speed), "vy": float(direction[1] * speed)},
            duration_s=0.2,
            include_cameras=False,
        )
        if result.get("terminated"):
            return bool(result.get("success"))
        elapsed += 0.2
    return float(np.linalg.norm(np.asarray(target_xy) - end_effector(ctx.observe(include_cameras=False))[:2])) <= tol * 1.5


def move_xy(ctx, target_xy: np.ndarray, *, squeeze: bool = False, timeout_s: float = 10.0, tol: float = 0.006) -> bool:
    elapsed = 0.0
    while elapsed < timeout_s:
        delta = np.asarray(target_xy, dtype=float) - end_effector(ctx.observe(include_cameras=False))[:2]
        distance = float(np.linalg.norm(delta))
        if distance <= tol:
            return True
        speed = min(SPEED, max(distance / 0.2, 0.015))
        direction = delta / distance
        action = {"vx": float(direction[0] * speed), "vy": float(direction[1] * speed)}
        if squeeze:
            action["gripper_delta"] = -0.05
        result = ctx.step(action, duration_s=0.2, include_cameras=False)
        if result.get("terminated"):
            return bool(result.get("success"))
        elapsed += 0.2
    return False


def velocity_to(
    ctx,
    target: np.ndarray,
    *,
    timeout_s: float,
    tol: float = 0.015,
    squeeze: float = 0.0,
) -> bool:
    elapsed = 0.0
    while elapsed < timeout_s:
        obs = ctx.observe(include_cameras=False)
        delta = np.asarray(target, dtype=float) - end_effector(obs)
        if float(np.linalg.norm(delta)) <= tol:
            return True
        norm = float(np.linalg.norm(delta)) + 1e-6
        speed = min(SPEED, max(norm / 0.25, 0.015))
        action = {
            "vx": float(delta[0] / norm * speed),
            "vy": float(delta[1] / norm * speed),
            "vz": float(delta[2] / norm * speed),
        }
        if squeeze:
            action["gripper_delta"] = float(squeeze)
        result = ctx.step(action, duration_s=0.2, include_cameras=False)
        if result.get("terminated"):
            return bool(result.get("success"))
        elapsed += 0.2
    return float(np.linalg.norm(np.asarray(target) - end_effector(ctx.observe(include_cameras=False)))) <= tol * 1.5


def slide(ctx, direction: np.ndarray, distance_m: float, *, squeeze: bool = False) -> None:
    direction = np.asarray(direction, dtype=float)
    direction = direction / (np.linalg.norm(direction) + 1e-9)
    start = end_effector(ctx.observe(include_cameras=False))[:2]
    elapsed = 0.0
    limit = distance_m / SPEED * 2.0 + 0.4
    while elapsed < limit:
        ee = end_effector(ctx.observe(include_cameras=False))[:2]
        if float(np.dot(ee - start, direction)) >= distance_m:
            return
        action = {"vx": float(direction[0] * SPEED), "vy": float(direction[1] * SPEED)}
        if squeeze:
            action["gripper_delta"] = -0.05
        result = ctx.step(action, duration_s=0.2, include_cameras=False)
        if result.get("terminated"):
            return
        elapsed += 0.2
