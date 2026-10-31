"""Damped Jacobian IK using the robot model and observed joints only."""

from __future__ import annotations

import mujoco as mj
import numpy as np

from .kinematics import _ARM, _ensure, end_effector

ARM5 = _ARM[:5]
WRIST_GRASP = np.array([1.62, 0.17], dtype=float)
HOME = np.array([0.0, -0.3, 0.6, 1.2, 0.0], dtype=float)
STEP_S = 1.0 / 30.0


def _q_from_obs(observation: dict) -> np.ndarray:
    return np.array([float(observation["joints"][name]["position_rad"]) for name in ARM5], dtype=float)


def _clip(q: np.ndarray) -> np.ndarray:
    model, _data = _ensure()
    out = q.copy()
    for index, name in enumerate(ARM5):
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        low, high = model.jnt_range[jid]
        out[index] = float(np.clip(out[index], low, high))
    return out


def _send(ctx, q: np.ndarray, gripper: float) -> bool:
    action = {f"{name}.pos": float(q[i]) for i, name in enumerate(ARM5)}
    action["gripper.pos"] = float(gripper)
    result = ctx.step(action, duration_s=STEP_S, include_cameras=False)
    return bool(result.get("terminated"))


def go_home(ctx) -> None:
    q = HOME.copy()
    for _ in range(90):
        if _send(ctx, q, 0.85):
            return
        live = _q_from_obs(ctx.observe(include_cameras=False))
        if float(np.max(np.abs(live - q))) < 0.05:
            return


def set_gripper(ctx, gripper: float, steps: int = 28) -> None:
    q = _q_from_obs(ctx.observe(include_cameras=False))
    start = float(ctx.observe(include_cameras=False)["joints"]["gripper"]["position_rad"])
    for index in range(steps):
        alpha = (index + 1) / steps
        if _send(ctx, q, (1.0 - alpha) * start + alpha * gripper):
            return


def servo_to(
    ctx,
    target: np.ndarray,
    *,
    gripper: float,
    steps: int = 160,
    xy_tol: float = 0.008,
    z_tol: float = 0.008,
    lock_wrist: bool = True,
    wrist: np.ndarray | None = None,
) -> bool:
    """Integrate commanded q; measure error from live observed EE."""
    model, data = _ensure()
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "gripperframe")
    q = _q_from_obs(ctx.observe(include_cameras=False))
    hold = None
    if lock_wrist:
        hold = WRIST_GRASP if wrist is None else np.asarray(wrist, dtype=float)
        q[3:] = hold
        for _ in range(24):
            if _send(ctx, q, gripper):
                return False
    arm_names = ARM5[:3] if lock_wrist else ARM5
    cols = [model.jnt_dofadr[mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)] for name in arm_names]
    for _ in range(steps):
        obs = ctx.observe(include_cameras=False)
        live_q = _q_from_obs(obs)
        # Jacobian at the live configuration.
        for name, value in zip(ARM5, live_q):
            jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
            data.qpos[model.jnt_qposadr[jid]] = float(value)
        mj.mj_forward(model, data)
        position = end_effector(obs)
        error = np.asarray(target, dtype=float) - position
        if float(np.linalg.norm(error[:2])) <= xy_tol and abs(float(error[2])) <= z_tol:
            _send(ctx, q, gripper)
            return True
        command = 2.2 * error
        norm = float(np.linalg.norm(command))
        cap = min(0.012, 0.55 * float(np.linalg.norm(error)))
        if norm > cap > 0:
            command = command * (cap / norm)
        jacobian = np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, jacobian, None, sid)
        j = jacobian[:, cols]
        dq = j.T @ np.linalg.solve(j @ j.T + 1e-3 * np.eye(3), command)
        full = np.zeros(5)
        full[: len(cols)] = dq
        trial = _clip(q + np.clip(full, -0.55, 0.55) * 0.05)
        if hold is not None:
            trial[3:] = hold
        q = trial
        if _send(ctx, q, gripper):
            return False
    return False
