#!/usr/bin/env python3
"""Yaw-aware scripted expert (no LLM, no ACT) on Lab 01 simple_pick.

Uses privileged cube pose to align the wrist with a cube face, approaches from
a face-normal standoff, then closes, verifies lift, and places.

The same entrypoint is the reusable collector: an existing LeRobot v3 root is
resumed (not wiped), yaw-bin and 2×2 XY-cell success counts are read from
``spawns.jsonl``, and ``--target-successes`` fills whatever is still short.

Visual smoke::

    ./labs/lab01_pnp/auto_collect_yaw.cmd

First 50 successes (yaw + XY quotas)::

    LAB01_YAW_RECORD=1 LAB01_YAW_TARGET_SUCCESSES=50 ./labs/lab01_pnp/auto_collect_yaw.cmd

Top-up the same dataset to 100 (empty low-Y cells get the extra 50)::

    LAB01_YAW_RECORD=1 LAB01_YAW_TARGET_SUCCESSES=100 ./labs/lab01_pnp/auto_collect_yaw.cmd
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Import LeRobot *before* mujoco — otherwise OSMesa+torch import order can SIGSEGV.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

import mujoco as mj
import numpy as np

from simstudio.common.eval_success import CONTAINER_XY, check_pick_success
from simstudio.common.lab01_spawn_quota import (
    XY_CELL_NAMES,
    XY_CELLS,
    XY_REMAINDER_ORDER,
    YAW_REMAINDER_ORDER,
    next_deficit,
    split_quota,
    xy_cell,
)
from simstudio.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot

ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Teleop elbow-up basin — already at grasp height (gf.z ≈ 0.006).
Q_READY = np.array([0.40, 0.99, -1.11, 1.56, 0.17], dtype=float)
WRIST_GRASP = np.array([1.62, 0.17], dtype=float)
GRIPPER_OPEN = 0.85
GRIPPER_CLOSED = -0.08
GRASP_Z = 0.012
LIFT_Z = 0.12
PLACE_Z = 0.08
KP = 2.2
LAM = 1e-3
DQ_LIM = 0.55
XY_TOL = 0.005
# Stage A: stop before the near face (cube half ≈ 15 mm).
XY_TOL_STANDOFF = 0.028
# Stage B: soft wrap. Slightly looser than the old 5 mm center-drive so the
# fixed jaw stops short of plowing through; no polish chase afterward.
XY_TOL_CREEP = 0.006
Z_TOL = 0.008
# Optional contact abort (off during creep — first tip hit is ~17 mm out).
CONTACT_MOVE_M = 0.012
PRE_CLOSE_SLIP_MAX_M = 0.008
CUBE_HALF_M = 0.015
FACE_MAX_OFF_APPROACH_DEG = 75.0
# Ignore early corner grazes until gripperframe is this close to the spawn center.
FIRST_CONTACT_REACH_M = 0.018
# After creep, recenter if the fingertip sits beside the face (far-reach drift).
PRE_CLOSE_TANGENT_MAX_M = 0.010
PRE_CLOSE_NORMAL_MAX_M = 0.020
# Both jaws must press the block this hard after closing, or the lift is skipped.
GRASP_CONTACT_MIN_N = 1.0
STANDOFF_MARGIN_M = 0.020
WRIST_ALIGN_TOL_RAD = 0.10
YAW_BIN_EDGES_DEG = np.linspace(-45.0, 45.0, 7)
TARGETED_MISS_GRIPPER = 0.50
# Directions, distances, and wrist errors seen when ACT closes but fails to lift.
DIVERSE_MISS_AXES = ("tangent", "radial", "diagonal")
DIVERSE_MISS_DISTANCES_M = (0.008, 0.012, 0.016)
DIVERSE_MISS_ROLLS_DEG = (5.0, -5.0, 10.0, -10.0)
DT = 0.05
STEP_SLEEP_S = float(os.environ.get("LAB01_YAW_SLEEP", "0.03"))
# Close/hold: keep one crisp squeeze, but don't linger (old 70+25 felt sluggish).
GRIPPER_SQUEEZE_STEPS = 28
GRIPPER_HOLD_STEPS = 8
TASK = "Pick up the cube and place it in the box"

# Optional per-tick recorder hook installed by main() when --record is set.
_frame_hook: Callable[[dict[str, Any], dict[str, float]], None] | None = None


class ScriptedRecorder:
    """Write scripted position-control ticks into a LeRobot v3 dataset."""

    def __init__(
        self,
        robot: SO101MujocoRobot,
        *,
        repo_id: str,
        root: Path,
        task: str = TASK,
        save_failures: bool = False,
    ) -> None:
        self.robot = robot
        self.task = task
        self.save_failures = save_failures
        self.root = Path(root)
        self.spawn_log = self.root / "spawns.jsonl"
        obs_ft = hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=True)
        act_ft = hw_to_dataset_features(robot.action_features, ACTION, use_video=True)
        features = {**obs_ft, **act_ft}
        info = self.root / "meta" / "info.json"
        if info.is_file():
            self.dataset = LeRobotDataset.resume(
                repo_id,
                root=self.root,
                tolerance_s=1e-3,
                image_writer_threads=4,
            )
            print(f"  resume dataset {repo_id} @ {self.root}", flush=True)
        else:
            # LeRobot create uses mkdir(exist_ok=False); wipe incomplete roots.
            if self.root.exists():
                shutil.rmtree(self.root)
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=int(robot.config.record_fps),
                features=features,
                root=self.root,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
            print(f"  create dataset {repo_id} @ {self.root}", flush=True)
        self.features = features
        self._ep_frames = 0
        self._attempt_meta: dict[str, Any] = {}
        self.saved = 0
        self.skipped = 0

    def begin_episode(self, attempt_meta: dict[str, Any]) -> None:
        self._attempt_meta = dict(attempt_meta)
        self._ep_frames = 0

    def update_metadata(self, **values: Any) -> None:
        self._attempt_meta.update(values)

    def on_tick(self, obs: dict[str, Any], action: dict[str, float]) -> None:
        obs_frame = build_dataset_frame(self.features, obs, prefix=OBS_STR)
        act_frame = build_dataset_frame(self.features, action, prefix=ACTION)
        self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})
        self._ep_frames += 1

    def end_episode(self, *, success: bool, episode: int) -> None:
        meta: dict[str, Any] = {
            "episode": episode,
            "success": success,
            "frames": self._ep_frames,
            **self._attempt_meta,
        }
        with self.spawn_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        if success or self.save_failures:
            if self._ep_frames > 0:
                self.dataset.save_episode()
                self.saved += 1
                print(
                    f"  saved episode frames={self._ep_frames} success={success} total_saved={self.saved}",
                    flush=True,
                )
            else:
                self.skipped += 1
        else:
            self._discard_episode_buffer()
            self.skipped += 1
            print(
                f"  discard failed episode frames={self._ep_frames} (set --save-failures to keep)",
                flush=True,
            )

    def _discard_episode_buffer(self) -> None:
        """clear_episode_buffer() plus removal of temp PNGs for video features.

        LeRobot v0.6 only deletes temp dirs of ``image`` features. Video PNGs of a
        discarded attempt stay on disk under the same episode index, and the next
        saved episode encodes the stale high-index frames as a trailing tail.
        """
        writer = self.dataset.writer
        if writer.episode_buffer is None:
            return
        if writer.image_writer is not None:
            writer._wait_image_writer()
        episode_index = int(np.asarray(writer.episode_buffer["episode_index"]).reshape(-1)[0])
        for key in self.dataset.meta.video_keys:
            img_dir = writer._get_image_file_dir(episode_index, key)
            if img_dir.is_dir():
                shutil.rmtree(img_dir)
        self.dataset.clear_episode_buffer()

    def close(self) -> None:
        # Flush any pending image writers.
        if hasattr(self.dataset, "finalize"):
            try:
                self.dataset.finalize()
            except Exception:
                pass
        print(
            f"recorder done: saved={self.saved} discarded={self.skipped} root={self.root}",
            flush=True,
        )


def _site_id(robot: SO101MujocoRobot) -> int:
    sid = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_SITE, "gripperframe")
    return sid if sid >= 0 else robot.ee_site_id


def _site_pos(robot: SO101MujocoRobot) -> np.ndarray:
    return np.asarray(robot.data.site_xpos[_site_id(robot)], dtype=float).copy()


def _jaw_contact_forces(robot: SO101MujocoRobot) -> tuple[float, float]:
    """Summed normal force (N) of the block on the fixed jaw and on the moving jaw."""
    m, d = robot.model, robot.data
    block = mj.mj_name2id(m, mj.mjtObj.mjOBJ_GEOM, "block")
    fixed = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "gripper")
    moving = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, "moving_jaw_so101_v1")
    f = np.zeros(6)
    out = [0.0, 0.0]
    for i in range(d.ncon):
        c = d.contact[i]
        if block not in (c.geom1, c.geom2):
            continue
        other = c.geom2 if c.geom1 == block else c.geom1
        body = m.geom_bodyid[other]
        if body not in (fixed, moving):
            continue
        mj.mj_contactForce(m, d, i, f)
        out[0 if body == fixed else 1] += abs(float(f[0]))
    return out[0], out[1]


def _title(robot: SO101MujocoRobot, text: str) -> None:
    win = getattr(robot, "_glfw_window", None)
    if win is None:
        return
    try:
        import glfw

        glfw.set_window_title(win, text)
    except Exception:
        pass


def _clip_arm(robot: SO101MujocoRobot, q: np.ndarray) -> np.ndarray:
    out = q.copy()
    for i, name in enumerate(ARM):
        lo, hi = robot._mujoco_joint_range[name]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def _q_arm(robot: SO101MujocoRobot) -> np.ndarray:
    return np.array([float(robot.data.ctrl[robot.act_ids[n]]) for n in ARM], dtype=float)


def _block_pose(robot: SO101MujocoRobot) -> tuple[float, float, float, float]:
    """Return privileged block XYZ + world yaw in degrees."""
    jid = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_JOINT, "block")
    if jid < 0:
        raise RuntimeError("block freejoint not found")
    adr = int(robot.model.jnt_qposadr[jid])
    x, y, z = (float(v) for v in robot.data.qpos[adr : adr + 3])
    w, qx, qy, qz = (float(v) for v in robot.data.qpos[adr + 3 : adr + 7])
    yaw = np.degrees(np.arctan2(2.0 * (w * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz)))
    return x, y, z, float(yaw)


def _square_angle_error(a: float, b: float) -> float:
    """Smallest angular error for a square (period pi/2)."""
    return abs(float((a - b + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0))


def _line_angle_error(a: float, b: float) -> float:
    """Smallest angular error between two undirected axes (period pi)."""
    return abs(float((a - b + np.pi / 2.0) % np.pi - np.pi / 2.0))


def _yaw_bin(yaw_deg: float, bins: int = 6) -> int:
    return min(bins - 1, max(0, int((yaw_deg + 45.0) / 90.0 * bins)))


def _xy_cell(x: float, y: float) -> str:
    return xy_cell(x, y)


def _wrist_roll_for_cube(
    robot: SO101MujocoRobot, cube_yaw_deg: float, face_normal: np.ndarray
) -> tuple[float, float]:
    """Numerically choose wrist roll whose gripper-frame y axis is perpendicular to ``face_normal``.

    Looking at the resulting world orientation is safer than assuming that
    wrist_roll equals world yaw: shoulder_pan and the site quaternion both
    contribute. Only the two rolls 180 degrees apart are equivalent: with the
    axis parallel to the approach normal the jaws hit the face side by side
    and close on air.
    """
    wrist_jid = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_JOINT, "wrist_roll")
    wrist_qpos = int(robot.model.jnt_qposadr[wrist_jid])
    q_saved = float(robot.data.qpos[wrist_qpos])
    lo, hi = robot._mujoco_joint_range["wrist_roll"]
    del cube_yaw_deg
    target = float(np.arctan2(face_normal[1], face_normal[0])) + np.pi / 2.0
    current = float(robot.data.ctrl[robot.act_ids["wrist_roll"]])
    candidates: list[tuple[float, float, float]] = []
    sid = _site_id(robot)
    for roll in np.linspace(lo, hi, 241):
        robot.data.qpos[wrist_qpos] = roll
        mj.mj_forward(robot.model, robot.data)
        rot = np.asarray(robot.data.site_xmat[sid], dtype=float).reshape(3, 3)
        axis = rot[:, 1]
        axis_yaw = float(np.arctan2(axis[1], axis[0]))
        err = _line_angle_error(axis_yaw, target)
        candidates.append((err, abs(float(roll) - current), float(roll)))
    robot.data.qpos[wrist_qpos] = q_saved
    mj.mj_forward(robot.model, robot.data)
    min_err = min(v[0] for v in candidates)
    # A square has four equivalent face axes. Numerical grid error differs by
    # tiny amounts; among equally aligned solutions, take the shortest joint
    # move instead of an unnecessary ~90-degree wrist turn.
    viable = [v for v in candidates if v[0] <= min_err + 0.01]
    best = min(viable, key=lambda v: v[1])
    return best[2], best[0]


def _face_standoff(
    cube_xy: np.ndarray, cube_yaw_deg: float, from_xy: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Pick the cube face nearest the gripper and return standoff + face normal."""
    yaw = np.radians(cube_yaw_deg)
    ux = np.array([np.cos(yaw), np.sin(yaw)])
    uy = np.array([-np.sin(yaw), np.cos(yaw)])
    normals = (ux, -ux, uy, -uy)
    toward_gripper = from_xy - cube_xy
    normal = max(normals, key=lambda n: float(np.dot(n, toward_gripper)))
    standoff = cube_xy + normal * (CUBE_HALF_M + STANDOFF_MARGIN_M)
    return standoff, normal


def _grasp_face_and_roll(
    robot: SO101MujocoRobot, cube_xy: np.ndarray, cube_yaw_deg: float
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Pick approach face + wrist roll for a centered jaw close.

    Ranking among the two gripper-facing face pairs:
    1. drop faces more than FACE_MAX_OFF_APPROACH_DEG off the approach,
    2. prefer wrist roll near WRIST_GRASP (rolls ≳1.1 rad close on air),
    3. prefer the nearer standoff (far-table faces sit past the arm's reach).
    """
    yaw = np.radians(cube_yaw_deg)
    ux = np.array([np.cos(yaw), np.sin(yaw)])
    uy = np.array([-np.sin(yaw), np.cos(yaw)])
    toward = _site_pos(robot)[:2] - cube_xy
    toward = toward / (np.linalg.norm(toward) + 1e-9)
    min_cos = float(np.cos(np.radians(FACE_MAX_OFF_APPROACH_DEG)))
    options = []
    for axis in (ux, uy):
        normal = axis if float(axis @ toward) >= 0.0 else -axis
        roll, err = _wrist_roll_for_cube(robot, cube_yaw_deg, normal)
        facing = float(normal @ toward)
        standoff = cube_xy + normal * (CUBE_HALF_M + STANDOFF_MARGIN_M)
        reach = float(np.linalg.norm(standoff))
        options.append(
            (facing < min_cos, abs(roll - float(WRIST_GRASP[1])), reach, -facing, normal, roll, err)
        )
    options.sort(key=lambda o: (o[0], o[1], o[2], o[3]))
    *_rank, normal, roll, err = options[0]
    return cube_xy + normal * (CUBE_HALF_M + STANDOFF_MARGIN_M), normal, roll, err


def _send_q(robot: SO101MujocoRobot, q_arm: np.ndarray, gripper: float) -> None:
    action = {f"{name}.pos": float(q_arm[i]) for i, name in enumerate(ARM)}
    lo, hi = robot._mujoco_joint_range["gripper"]
    action["gripper.pos"] = float(np.clip(gripper, lo, hi))
    obs = robot.get_observation() if _frame_hook is not None else None
    sent = robot.send_action(action)
    if _frame_hook is not None and obs is not None:
        _frame_hook(obs, sent)
    if STEP_SLEEP_S > 0:
        time.sleep(STEP_SLEEP_S)


def goto_q(
    robot: SO101MujocoRobot,
    q_goal: np.ndarray,
    *,
    gripper: float,
    steps: int,
    phase: str,
) -> None:
    _title(robot, f"SO-101 scripted PoC — {phase}")
    print(f"  phase {phase}", flush=True)
    q0 = _q_arm(robot)
    for i in range(steps):
        a = (i + 1) / steps
        a = a * a * (3.0 - 2.0 * a)
        _send_q(robot, _clip_arm(robot, (1.0 - a) * q0 + a * q_goal), gripper)


def goto_ready_without_passing_cube(
    robot: SO101MujocoRobot,
    cube_xy: np.ndarray,
    *,
    gripper: float,
    steps: int = 50,
) -> None:
    """Move toward Q_READY with distance-based deceleration; settle by the cube.

    Full joint interp to ``Q_READY`` would swing past the cube. We follow that
    path but slow down as ``d_cube`` shrinks, then freeze at the current pose
    (no snap-back to an earlier joint state — that felt like a hard brake).
    """
    _title(robot, "SO-101 scripted PoC — ready")
    print("  phase ready (decelerate into cube approach)", flush=True)
    q0 = _q_arm(robot)
    q_goal = Q_READY.copy()
    # Begin easing below this distance; settle around historical closest (~3.2 cm).
    slow_start_m = 0.10
    soft_stop_m = 0.036
    a = 0.0
    prev_d = float(np.linalg.norm(_site_pos(robot)[:2] - cube_xy))
    # Extra ticks allowed because we crawl near the cube.
    for i in range(steps * 3):
        p = _site_pos(robot)
        d = float(np.linalg.norm(p[:2] - cube_xy))
        passed = float(p[0]) > float(cube_xy[0]) + 0.003
        close_enough = d <= soft_stop_m
        # Started receding while already in the slow zone — settle here.
        receding = i > 8 and d < slow_start_m and d > prev_d + 0.0015

        if close_enough or passed or receding:
            q_now = _q_arm(robot)
            for _ in range(10):
                _send_q(robot, q_now, gripper)
            why = "soft-stop" if close_enough else ("passed +X" if passed else "receding")
            print(
                f"    ready-settle ({why}) gf={np.round(p, 4)} d_cube={d:.4f}",
                flush=True,
            )
            return

        # Speed scale: 1 far away → ~0.1 at soft_stop (quadratic ease).
        if d >= slow_start_m:
            scale = 1.0
        else:
            t = (d - soft_stop_m) / (slow_start_m - soft_stop_m)
            t = float(np.clip(t, 0.0, 1.0))
            scale = 0.10 + 0.90 * (t * t)

        da = (1.0 / float(steps)) * scale
        a = min(1.0, a + da)
        s = a * a * (3.0 - 2.0 * a)
        q = _clip_arm(robot, (1.0 - s) * q0 + s * q_goal)
        _send_q(robot, q, gripper)
        prev_d = d
        if a >= 1.0 - 1e-6:
            break

    print(
        f"    ready-end gf={np.round(_site_pos(robot), 4)} "
        f"d_cube={float(np.linalg.norm(_site_pos(robot)[:2] - cube_xy)):.4f}",
        flush=True,
    )


def servo_xyz(
    robot: SO101MujocoRobot,
    target: np.ndarray,
    *,
    gripper: float,
    steps: int,
    phase: str,
    lock_wrist: bool = True,
    wrist_target: np.ndarray | None = None,
    xy_tol: float = XY_TOL,
    z_tol: float = Z_TOL,
    kp: float = KP,
    dq_lim: float = DQ_LIM,
    max_step: float = 0.012,
    prevent_overshoot: bool = True,
    stop_on_cube_move: bool = False,
    contact_move_m: float = CONTACT_MOVE_M,
    cube_ref: tuple[float, float, float] | None = None,
    contact_reach_m: float | None = None,
) -> str:
    """Servo gripperframe toward target.

    With ``prevent_overshoot``, each step is trial-applied and **rejected** if
    Cartesian error grows or the site crosses to the far side of the target
    (IK often moves farther than the capped command — that was the visible
    overshoot-then-return).

    Returns ``reached``, ``contact``, or ``timeout``.
    """
    _title(robot, f"SO-101 scripted PoC — {phase}")
    if not phase.startswith("_"):
        print(f"  phase {phase} → {np.round(target, 4)}", flush=True)
    n_arm = 3 if lock_wrist else 5
    if wrist_target is None:
        wrist_target = WRIST_GRASP
    cols = [robot.dof_ids[n] for n in ARM[:n_arm]]
    sid = _site_id(robot)
    q = _q_arm(robot)
    if lock_wrist:
        q[3:] = wrist_target
    if cube_ref is None:
        cube_ref = robot.get_block_position()
    best = float("inf")
    best_p = _site_pos(robot)
    for _ in range(steps):
        mj.mj_forward(robot.model, robot.data)
        if stop_on_cube_move and cube_ref is not None:
            cube_now = robot.get_block_position()
            if cube_now is not None:
                moved = float(np.linalg.norm(np.asarray(cube_now[:2]) - np.asarray(cube_ref[:2])))
                reach = float(np.linalg.norm(_site_pos(robot)[:2] - np.asarray(cube_ref[:2])))
                near_enough = contact_reach_m is None or reach <= contact_reach_m
                if moved >= contact_move_m and near_enough:
                    for _hold in range(8):
                        _send_q(robot, q, gripper)
                    moved = float(
                        np.linalg.norm(np.asarray(robot.get_block_position()[:2]) - np.asarray(cube_ref[:2]))
                    )
                    print(
                        f"    contact: cube moved {moved * 1000:.1f} mm — freeze & close",
                        flush=True,
                    )
                    return "contact"
        p = _site_pos(robot)
        err = target - p
        err_n = float(np.linalg.norm(err))
        xy_err = float(np.linalg.norm(err[:2]))
        if err_n < best:
            best, best_p = err_n, p.copy()
        if xy_err < xy_tol and abs(err[2]) < z_tol:
            _send_q(robot, q, gripper)
            print(f"    reached xyz={np.round(p, 4)} err={err_n:.4f}", flush=True)
            return "reached"

        scale = 1.0
        if prevent_overshoot and err_n < 0.02:
            scale = max(0.35, err_n / 0.02)
        cmd = kp * scale * err
        cmd_n = float(np.linalg.norm(cmd))
        if prevent_overshoot:
            # Stay well short of the remaining error so Jacobian slack can't
            # carry us through the target in one tick.
            cap = min(max_step, 0.55 * err_n) if err_n > 1e-9 else max_step
        else:
            cap = max_step
        if cmd_n > cap:
            cmd = cmd * (cap / cmd_n)

        jp = np.zeros((3, robot.model.nv))
        mj.mj_jacSite(robot.model, robot.data, jp, None, sid)
        j = jp[:, cols]
        dq_n = j.T @ np.linalg.solve(j @ j.T + LAM * np.eye(3), cmd)
        dq = np.zeros(5)
        dq[:n_arm] = dq_n
        q_trial = _clip_arm(robot, q + np.clip(dq, -dq_lim, dq_lim) * DT)
        if lock_wrist:
            q_trial[3:] = wrist_target

        if prevent_overshoot:
            q_saved = q.copy()
            _send_q(robot, q_trial, gripper)
            mj.mj_forward(robot.model, robot.data)
            p2 = _site_pos(robot)
            err2 = target - p2
            err2_n = float(np.linalg.norm(err2))
            # Only reject a true far-side crossing (past the target). Sideways
            # IK wobble that briefly grows error is allowed — rejecting it
            # was starving the approach of useful progress.
            crossed = float(np.dot(err[:2], err2[:2])) < 0.0
            if crossed:
                q = q_saved
                _send_q(robot, q, gripper)
                print(
                    f"    stop-short xyz={np.round(p, 4)} err={err_n:.4f} (blocked far-side cross)",
                    flush=True,
                )
                return "reached"
            q = q_trial
            if err2_n < best:
                best, best_p = err2_n, p2.copy()
        else:
            q = q_trial
            _send_q(robot, q, gripper)

    print(
        f"    timeout best={np.round(best_p, 4)} err={best:.4f} "
        f"now={np.round(_site_pos(robot), 4)} cube={robot.get_block_position()}",
        flush=True,
    )
    return "timeout"


def set_grip(robot: SO101MujocoRobot, g1: float, steps: int = 55) -> None:
    q = _q_arm(robot)
    g0 = float(robot.data.ctrl[robot.act_ids["gripper"]])
    _title(robot, "SO-101 scripted PoC — gripper")
    print(f"  phase gripper {g0:.2f} → {g1:.2f}", flush=True)
    for i in range(steps):
        a = (i + 1) / steps
        a = a * a * (3.0 - 2.0 * a)
        _send_q(robot, q, (1.0 - a) * g0 + a * g1)


def hold(robot: SO101MujocoRobot, gripper: float, steps: int = 15) -> None:
    q = _q_arm(robot)
    for _ in range(steps):
        _send_q(robot, q, gripper)


# Off-tube joint kicks applied at the face standoff, before the creep.
# wrist_roll stays 0: v4 already spreads it. wrist_flex std on v4 is ~0.09 rad,
# so ±0.30–0.45 is several times that tube. Shoulder and elbow kicks are large
# enough that the following IK correction is not the same approach every time.
JOINT_KICKS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.0, 0.55, 0.0, 0.0, 0.0),
    (0.0, -0.45, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.60, 0.0, 0.0),
    (0.0, 0.0, -0.55, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.40, 0.0),
    (0.0, 0.0, 0.0, -0.35, 0.0),
    (0.40, 0.0, 0.0, 0.0, 0.0),
    (-0.35, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.40, -0.45, 0.0, 0.0),
    (0.0, 0.0, 0.40, 0.30, 0.0),
    (0.0, -0.35, 0.0, -0.30, 0.0),
    (0.25, 0.40, -0.35, 0.0, 0.0),
    (-0.25, -0.40, 0.35, 0.25, 0.0),
    (0.0, 0.50, 0.35, -0.30, 0.0),
    (0.30, -0.35, -0.40, 0.35, 0.0),
)

# Off-tube kicks after a successful lift, before the scripted place. Magnitudes
# stay under the pre-grasp kicks so the cube can remain in the closed gripper.
# wrist_roll stays 0. The following place servo is the expert correction.
CARRY_KICKS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.0, 0.28, 0.0, 0.0, 0.0),
    (0.0, -0.25, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.30, 0.0, 0.0),
    (0.0, 0.0, -0.28, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.22, 0.0),
    (0.0, 0.0, 0.0, -0.22, 0.0),
    (0.22, 0.0, 0.0, 0.0, 0.0),
    (-0.20, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.22, -0.22, 0.0, 0.0),
    (0.15, 0.18, 0.0, -0.15, 0.0),
    (-0.15, -0.18, 0.18, 0.0, 0.0),
    (0.0, 0.20, 0.18, 0.18, 0.0),
)

# Kicks once the closed gripper is already over the container. The expert then
# lowers and opens, so the label covers the state just before release.
RELEASE_KICKS: tuple[tuple[float, float, float, float, float], ...] = (
    (0.0, 0.18, 0.0, 0.0, 0.0),
    (0.0, -0.16, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.18, 0.0, 0.0),
    (0.0, 0.0, -0.16, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.16, 0.0),
    (0.0, 0.0, 0.0, -0.16, 0.0),
    (0.16, 0.0, 0.0, 0.0, 0.0),
    (-0.14, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.14, -0.14, 0.0, 0.0),
    (0.12, 0.0, 0.0, -0.12, 0.0),
    (-0.12, -0.12, 0.12, 0.0, 0.0),
    (0.0, 0.12, 0.12, 0.12, 0.0),
)


def _apply_joint_kick(
    robot: SO101MujocoRobot,
    kick: np.ndarray,
    wrist_target: np.ndarray,
) -> np.ndarray:
    """Move off the nominal tube, then ease the wrist back. IK corrects the arm."""
    q = _q_arm(robot)
    q_kick = _clip_arm(robot, q + kick)
    applied = q_kick - q
    goto_q(robot, q_kick, gripper=GRIPPER_OPEN, steps=24, phase="joint-kick")
    q_wrist = _q_arm(robot)
    q_wrist[3] = float(wrist_target[0])
    q_wrist[4] = float(wrist_target[1])
    goto_q(robot, _clip_arm(robot, q_wrist), gripper=GRIPPER_OPEN, steps=16, phase="joint-kick-wrist-back")
    print(
        f"    joint-kick applied rad={np.round(applied, 3)} "
        f"q={np.round(_q_arm(robot), 3)}",
        flush=True,
    )
    return applied


def _apply_carry_kick(
    robot: SO101MujocoRobot,
    kick: np.ndarray,
    gripper: float,
) -> np.ndarray:
    """Shift the lifted arm off the carry tube without opening the gripper."""
    q = _q_arm(robot)
    q_kick = _clip_arm(robot, q + kick)
    applied = q_kick - q
    goto_q(robot, q_kick, gripper=gripper, steps=24, phase="carry-kick")
    print(
        f"    held-kick applied rad={np.round(applied, 3)} "
        f"cube={robot.get_block_position()}",
        flush=True,
    )
    return applied


def _attempt_lift(
    robot: SO101MujocoRobot,
    recorder: ScriptedRecorder | None,
    *,
    cx: float,
    cy: float,
    cz: float,
    cube_yaw_deg: float,
    spawn: tuple[float, float, float],
    close_dx: float,
    close_dy: float,
    close_droll: float,
    gripper_closed: float | None,
    phase_prefix: str = "",
    contact_reach_m: float | None = None,
    stop_before_close: bool = False,
    slip_max_m: float = PRE_CLOSE_SLIP_MAX_M,
    joint_kick: np.ndarray | None = None,
) -> dict[str, Any]:
    """Align, creep, close, and try to lift. Does not reset or end the episode."""
    prefix = f"{phase_prefix}" if phase_prefix else ""
    cube_xy = np.array([cx, cy], dtype=float)
    grasp = np.array([cx, cy, GRASP_Z])
    standoff_xy, face_normal, wrist_roll_target, align_err = _grasp_face_and_roll(
        robot, cube_xy, cube_yaw_deg
    )
    wrist_target = np.array([WRIST_GRASP[0], wrist_roll_target], dtype=float)
    q = _q_arm(robot)
    q[3:] = wrist_target
    goto_q(robot, q, gripper=GRIPPER_OPEN, steps=24, phase=f"{prefix}align-wrist")
    if recorder is not None and not phase_prefix:
        recorder.update_metadata(
            wrist_roll_target=float(wrist_roll_target),
            wrist_align_error_rad=float(align_err),
        )
    print(
        f"    wrist yaw={cube_yaw_deg:+.1f}° target_roll={wrist_roll_target:+.3f} "
        f"align_err={align_err:.4f} gf={np.round(_site_pos(robot), 4)}",
        flush=True,
    )

    status_standoff = servo_xyz(
        robot,
        np.array([standoff_xy[0], standoff_xy[1], GRASP_Z]),
        gripper=GRIPPER_OPEN,
        steps=70,
        phase=f"{prefix}face-standoff",
        wrist_target=wrist_target,
        xy_tol=0.008,
        kp=1.8,
        dq_lim=0.30,
        max_step=0.012,
        prevent_overshoot=True,
    )
    if recorder is not None and not phase_prefix:
        recorder.update_metadata(
            face_normal=[float(face_normal[0]), float(face_normal[1])],
            standoff_xy=[float(standoff_xy[0]), float(standoff_xy[1])],
            standoff_status=status_standoff,
        )

    if joint_kick is not None and float(np.linalg.norm(joint_kick)) > 0.0:
        applied = _apply_joint_kick(robot, np.asarray(joint_kick, dtype=float), wrist_target)
        if recorder is not None and not phase_prefix:
            recorder.update_metadata(joint_kick=[float(v) for v in applied])

    status = servo_xyz(
        robot,
        grasp,
        gripper=GRIPPER_OPEN,
        steps=70,
        phase=f"{prefix}creep",
        wrist_target=wrist_target,
        xy_tol=XY_TOL_CREEP,
        kp=1.7,
        dq_lim=0.28,
        max_step=0.015,
        prevent_overshoot=False,
        stop_on_cube_move=True,
        cube_ref=spawn,
        contact_move_m=0.003,
        contact_reach_m=contact_reach_m,
    )
    print(
        f"    after-creep status={status} gf={np.round(_site_pos(robot), 4)} "
        f"cube={robot.get_block_position()}",
        flush=True,
    )

    # Near-miss lateral offset: tip is close but beside the face. Far out-of-reach
    # poses (xy_err ≳ 2 cm) must not polish — restandoff just shoves the cube.
    cube_now = robot.get_block_position()
    gf = _site_pos(robot)
    xy_err = float(np.linalg.norm(gf[:2] - np.asarray(spawn[:2])))
    tan_err = 0.0
    nor_err = 0.0
    if cube_now is not None:
        rel = gf[:2] - np.asarray(cube_now[:2])
        tangent = np.array([-face_normal[1], face_normal[0]])
        tan_err = float(rel @ tangent)
        nor_err = float(rel @ face_normal)
        if xy_err <= 0.018 and (
            abs(tan_err) > PRE_CLOSE_TANGENT_MAX_M or nor_err > PRE_CLOSE_NORMAL_MAX_M
        ):
            standoff = np.asarray(cube_now[:2]) + face_normal * (CUBE_HALF_M + STANDOFF_MARGIN_M)
            print(
                f"    pre-close polish tangent={tan_err * 1000:+.1f}mm "
                f"normal={nor_err * 1000:+.1f}mm → restandoff+creep",
                flush=True,
            )
            servo_xyz(
                robot,
                np.array([standoff[0], standoff[1], GRASP_Z]),
                gripper=GRIPPER_OPEN,
                steps=40,
                phase=f"{prefix}polish-standoff",
                wrist_target=wrist_target,
                xy_tol=0.008,
                kp=1.6,
                dq_lim=0.25,
                max_step=0.010,
                prevent_overshoot=True,
            )
            cube_now = robot.get_block_position() or cube_now
            status = servo_xyz(
                robot,
                np.array([cube_now[0], cube_now[1], GRASP_Z]),
                gripper=GRIPPER_OPEN,
                steps=50,
                phase=f"{prefix}polish-creep",
                wrist_target=wrist_target,
                xy_tol=XY_TOL_CREEP,
                kp=1.5,
                dq_lim=0.22,
                max_step=0.008,
                prevent_overshoot=False,
                stop_on_cube_move=True,
                cube_ref=cube_now,
                contact_move_m=0.003,
                contact_reach_m=0.014,
            )
            cube_now = robot.get_block_position() or cube_now
            cx, cy, cz = float(cube_now[0]), float(cube_now[1]), float(cube_now[2])
            gf = _site_pos(robot)
            rel = gf[:2] - np.asarray(cube_now[:2])
            tan_err = float(rel @ tangent)
            nor_err = float(rel @ face_normal)
            xy_err = float(np.linalg.norm(gf[:2] - np.asarray(spawn[:2])))

    cube_slip = (
        float(np.linalg.norm(np.asarray(cube_now[:2]) - np.asarray(spawn[:2])))
        if cube_now is not None
        else 0.0
    )
    print(
        f"    pre-close status={status} gf={np.round(gf, 4)} cube={cube_now} "
        f"xy_to_spawn={xy_err:.4f} cube_slip={cube_slip * 1000:.1f}mm "
        f"tangent={tan_err * 1000:+.1f}mm normal={nor_err * 1000:+.1f}mm",
        flush=True,
    )
    if recorder is not None and not phase_prefix:
        recorder.update_metadata(
            creep_status=status,
            pre_close_xy_error_m=xy_err,
            cube_slip_mm=cube_slip * 1000.0,
            wrist_roll_at_close=float(_q_arm(robot)[4]),
            pre_close_tangent_mm=tan_err * 1000.0,
            pre_close_normal_mm=nor_err * 1000.0,
        )
    empty = {
        "lifted": False,
        "rejected": True,
        "wrist_target": wrist_target,
        "face_normal": face_normal,
        "z_after_close": cz,
        "grip_cmd": GRIPPER_CLOSED,
    }
    if cube_slip > slip_max_m or align_err > WRIST_ALIGN_TOL_RAD:
        print(
            f"    quality reject before close: slip={cube_slip * 1000:.1f}mm align_err={align_err:.3f}rad",
            flush=True,
        )
        return empty
    # Normal can be 20–30 mm and still grasp (jaws extend past gripperframe).
    # A large tangent means the tip is beside the cube — closing there is empty.
    if abs(tan_err) > PRE_CLOSE_TANGENT_MAX_M:
        print(
            f"    quality reject before close: tangent={tan_err * 1000:+.1f}mm "
            f"normal={nor_err * 1000:+.1f}mm (beside face)",
            flush=True,
        )
        return empty

    grip_cmd = GRIPPER_CLOSED if gripper_closed is None else float(gripper_closed)
    if close_dx or close_dy or close_droll:
        wrist_target = wrist_target.copy()
        wrist_target[1] = float(wrist_target[1] + close_droll)
        servo_xyz(
            robot,
            np.array([cx + close_dx, cy + close_dy, GRASP_Z]),
            gripper=GRIPPER_OPEN,
            steps=40,
            phase=f"{prefix}probe-offset",
            wrist_target=wrist_target,
            xy_tol=0.004,
            kp=1.5,
            dq_lim=0.25,
            max_step=0.008,
            prevent_overshoot=True,
        )
        print(
            f"    probe-offset dx={close_dx * 1000:.1f}mm dy={close_dy * 1000:.1f}mm "
            f"droll={close_droll:+.3f}rad gf={np.round(_site_pos(robot), 4)}",
            flush=True,
        )

    if stop_before_close:
        print(f"    parked open before close gf={np.round(_site_pos(robot), 4)}", flush=True)
        return {
            "lifted": False,
            "rejected": False,
            "wrist_target": wrist_target,
            "face_normal": face_normal,
            "z_after_close": cz,
            "grip_cmd": GRIPPER_OPEN,
        }

    gf_close = _site_pos(robot)
    cube_before = np.asarray(robot.get_block_position() or (cx, cy, cz), dtype=float)
    set_grip(robot, grip_cmd, steps=GRIPPER_SQUEEZE_STEPS)
    hold(robot, grip_cmd, steps=GRIPPER_HOLD_STEPS)
    z_after_close = (robot.get_block_position() or (0.0, 0.0, 0.0))[2]
    cube_after = np.asarray(robot.get_block_position() or cube_before, dtype=float)
    f_fixed, f_moving = _jaw_contact_forces(robot)
    rel = gf_close - cube_before
    tangent = np.array([-face_normal[1], face_normal[0]])
    jaw_axis = np.asarray(robot.data.site_xmat[_site_id(robot)], dtype=float).reshape(3, 3)[:2, 1]
    jaw_vs_normal_deg = float(
        np.degrees(np.arccos(np.clip(abs(jaw_axis @ face_normal) / (np.linalg.norm(jaw_axis) + 1e-9), 0.0, 1.0)))
    )
    close_geom = {
        "f_fixed_n": f_fixed,
        "f_moving_n": f_moving,
        "jaw_vs_normal_deg": jaw_vs_normal_deg,
        "close_normal_m": float(rel[:2] @ face_normal),
        "close_tangent_m": float(rel[:2] @ tangent),
        "close_dz_m": float(rel[2]),
        "close_cube_shift_m": float(np.linalg.norm(cube_after[:2] - cube_before[:2])),
        "q_gripper_after_close": float(
            robot.data.qpos[robot.model.jnt_qposadr[mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_JOINT, "gripper")]]
        ),
    }
    print(
        f"    close-contact fixed={f_fixed:.2f}N moving={f_moving:.2f}N "
        f"normal={close_geom['close_normal_m'] * 1000:+.1f}mm tangent={close_geom['close_tangent_m'] * 1000:+.1f}mm "
        f"dz={close_geom['close_dz_m'] * 1000:+.1f}mm shift={close_geom['close_cube_shift_m'] * 1000:.1f}mm "
        f"q_grip={close_geom['q_gripper_after_close']:+.3f}",
        flush=True,
    )
    if min(f_fixed, f_moving) < GRASP_CONTACT_MIN_N:
        print(
            f"    no two-jaw contact (min {min(f_fixed, f_moving):.2f}N < {GRASP_CONTACT_MIN_N}N) — skip lift",
            flush=True,
        )
        return {
            "lifted": False,
            "rejected": False,
            "wrist_target": wrist_target,
            "face_normal": face_normal,
            "z_after_close": z_after_close,
            "grip_cmd": grip_cmd,
            "close_geom": close_geom,
        }
    hold_xy = _site_pos(robot)
    servo_xyz(
        robot,
        np.array([hold_xy[0], hold_xy[1], LIFT_Z]),
        gripper=grip_cmd,
        steps=100,
        phase=f"{prefix}lift",
        lock_wrist=True,
        wrist_target=wrist_target,
        prevent_overshoot=False,
        max_step=1.0,
    )
    lifted = (robot.get_block_position() or (0.0, 0.0, 0.0))[2] >= cz + 0.02
    print(f"    after-lift cube={robot.get_block_position()} lifted={lifted}", flush=True)
    return {
        "lifted": lifted,
        "rejected": False,
        "wrist_target": wrist_target,
        "face_normal": face_normal,
        "z_after_close": z_after_close,
        "grip_cmd": grip_cmd,
        "close_geom": close_geom,
    }


def _diverse_miss(
    episode: int,
    face_normal: np.ndarray,
    *,
    large_only: bool = False,
) -> tuple[float, float, float, dict[str, Any]]:
    """Choose a varied close-without-lift perturbation.

    The 36 combinations cover lateral, short/long, and diagonal misses at two
    wrist-error magnitudes. A stride coprime with 36 visits every combination
    before repeating.
    """
    normal = np.asarray(face_normal, dtype=float)
    norm = float(np.linalg.norm(normal))
    normal = np.array([0.0, -1.0]) if norm < 1e-6 else normal / norm
    tangent = np.array([-normal[1], normal[0]], dtype=float)
    variants = [
        (axis, distance, roll)
        for axis in DIVERSE_MISS_AXES
        for distance in DIVERSE_MISS_DISTANCES_M
        for roll in DIVERSE_MISS_ROLLS_DEG
    ]
    if large_only:
        variants = [item for item in variants if item[1] >= 0.012 and abs(item[2]) >= 10.0]
    axis, distance, roll_deg = variants[(episode * 7) % len(variants)]
    if axis == "tangent":
        direction = tangent
    elif axis == "radial":
        direction = normal if roll_deg > 0 else -normal
    else:
        direction = tangent + (normal if roll_deg > 0 else -normal)
        direction = direction / float(np.linalg.norm(direction))
    if roll_deg < 0 and axis == "tangent":
        direction = -direction
    delta = float(distance) * direction
    return (
        float(delta[0]),
        float(delta[1]),
        float(np.radians(roll_deg)),
        {
            "miss_axis": axis,
            "miss_distance_m": float(distance),
            "miss_roll_deg": float(roll_deg),
        },
    )


def _prepare_regrasp(
    robot: SO101MujocoRobot,
    cube_xy: np.ndarray,
    cube_yaw_deg: float,
) -> None:
    """Release, move above a safe standoff, then descend for reacquisition.

    The failed lift leaves the gripper high but in a joint basin from which a
    diagonal 3-DOF descent stalls.  A high horizontal move followed by a
    vertical descent keeps clear of the cube and returns to the same reachable
    low basin used by the nominal expert.
    """
    set_grip(robot, GRIPPER_OPEN, steps=24)
    standoff_xy, _normal = _face_standoff(cube_xy, cube_yaw_deg, _site_pos(robot)[:2])
    q_now = _q_arm(robot)
    wrist_target = q_now[3:5].copy()
    high_z = max(float(_site_pos(robot)[2]), 0.10)
    servo_xyz(
        robot,
        np.array([standoff_xy[0], standoff_xy[1], high_z]),
        gripper=GRIPPER_OPEN,
        steps=80,
        phase="regrasp-high-standoff",
        lock_wrist=True,
        wrist_target=wrist_target,
        prevent_overshoot=False,
        max_step=1.0,
    )
    servo_xyz(
        robot,
        np.array([standoff_xy[0], standoff_xy[1], GRASP_Z]),
        gripper=GRIPPER_OPEN,
        steps=140,
        phase="regrasp-descend",
        lock_wrist=True,
        wrist_target=wrist_target,
        prevent_overshoot=False,
        max_step=1.0,
    )


def run_episode(
    robot: SO101MujocoRobot,
    episode: int,
    *,
    random_cube: bool,
    recorder: ScriptedRecorder | None = None,
    close_dx: float = 0.0,
    close_dy: float = 0.0,
    close_droll: float = 0.0,
    gripper_closed: float | None = None,
    stop_after_lift: bool = False,
    start_jitter_rad: float = 0.0,
    allow_regrasp: bool = False,
    force_first_miss: bool = False,
    record_recovery_only: bool = False,
    recovery_style: str = "diverse",
    miss_gripper: float = TARGETED_MISS_GRIPPER,
    miss_large_only: bool = False,
    preclose: bool = False,
    joint_kick: np.ndarray | None = None,
    carry_kick: np.ndarray | None = None,
    release_kick: np.ndarray | None = None,
    drop_after_lift: bool = False,
) -> bool:
    global _frame_hook
    robot.reset_episode(episode)
    cx, cy, cz, cube_yaw_deg = _block_pose(robot)
    print(
        f"episode {episode} spawn=({cx:.3f}, {cy:.3f}, {cz:.3f}) "
        f"yaw={cube_yaw_deg:+.1f}° random={random_cube}",
        flush=True,
    )

    spawn = (cx, cy, cz)
    episode_meta = {
        "spawn": list(spawn),
        "yaw_deg": cube_yaw_deg,
        "yaw_bin": _yaw_bin(cube_yaw_deg),
        "xy_cell": _xy_cell(cx, cy),
    }
    if recorder is not None:
        recorder.begin_episode(episode_meta)
        if record_recovery_only:
            # Never train on deliberately wrong actions.  Recording starts
            # only after the perturbed grasp has demonstrably failed.
            _frame_hook = None
    cube_xy = np.array([cx, cy], dtype=float)

    # 1) Move toward elbow-up ready but DO NOT park past the cube.
    #    Full goto_q(Q_READY) swings home → near cube → far side; the old
    #    slow "approach" was just walking back from that overshoot.
    if start_jitter_rad > 0:
        q_home = _clip_arm(robot, _q_arm(robot))
        noise = np.random.uniform(-start_jitter_rad, start_jitter_rad, size=5)
        q_jit = q_home.copy()
        q_jit[:5] = q_jit[:5] + noise
        q_jit = _clip_arm(robot, q_jit)
        goto_q(robot, q_jit, gripper=GRIPPER_OPEN, steps=16, phase="start-jitter")
        if recorder is not None:
            recorder.update_metadata(start_jitter_rad=float(start_jitter_rad))
    goto_ready_without_passing_cube(robot, cube_xy, gripper=GRIPPER_OPEN, steps=50)
    grasp_kwargs = dict(
        close_dx=close_dx,
        close_dy=close_dy,
        close_droll=close_droll,
        gripper_closed=gripper_closed,
    )
    if preclose or force_first_miss:
        _standoff, face_normal = _face_standoff(
            np.array([cx, cy]), cube_yaw_deg, _site_pos(robot)[:2]
        )
        miss_dx, miss_dy, miss_droll, miss_meta = _diverse_miss(
            episode, face_normal, large_only=miss_large_only
        )
        print(
            f"    targeted first grasp: {miss_meta['miss_axis']} "
            f"{miss_meta['miss_distance_m'] * 1000:.0f}mm "
            f"shift=({miss_dx * 1000:+.1f}, {miss_dy * 1000:+.1f})mm "
            f"roll={miss_meta['miss_roll_deg']:+.0f}°",
            flush=True,
        )
    if preclose:
        _attempt_lift(
            robot,
            None,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=float(miss_dx),
            close_dy=float(miss_dy),
            close_droll=float(miss_droll),
            gripper_closed=GRIPPER_OPEN,
            stop_before_close=True,
        )
        if recorder is not None and record_recovery_only:
            recorder.begin_episode(
                {
                    **episode_meta,
                    "targeted_recovery": True,
                    "recovery_style": recovery_style,
                    "first_miss_shift_xy_m": [float(miss_dx), float(miss_dy)],
                    "first_miss_roll_rad": float(miss_droll),
                    **miss_meta,
                }
            )
            _frame_hook = recorder.on_tick
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        spawn = (cx, cy, cz)
        attempt = _attempt_lift(
            robot,
            recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=0.0,
            close_dy=0.0,
            close_droll=0.0,
            gripper_closed=None,
            phase_prefix="correct-",
            contact_reach_m=0.012,
        )
    elif drop_after_lift:
        attempt = _attempt_lift(
            robot,
            None,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=0.0,
            close_dy=0.0,
            close_droll=0.0,
            gripper_closed=None,
            contact_reach_m=FIRST_CONTACT_REACH_M,
        )
        if not attempt["lifted"]:
            print("    drop-carry: nominal lift failed — skip", flush=True)
            return False
        site = _site_pos(robot)
        mid_xy = 0.5 * (site[:2] + np.asarray(CONTAINER_XY, dtype=float))
        servo_xyz(
            robot,
            np.array([mid_xy[0], mid_xy[1], LIFT_Z]),
            gripper=GRIPPER_CLOSED,
            steps=50,
            phase="drop-carry-mid",
            lock_wrist=True,
            wrist_target=attempt["wrist_target"],
            prevent_overshoot=False,
            max_step=1.0,
        )
        set_grip(robot, GRIPPER_OPEN, steps=24)
        hold(robot, GRIPPER_OPEN, steps=30)
        dropped = robot.get_block_position()
        dropped_z = float(dropped[2]) if dropped is not None else 0.0
        print(f"    drop-carry released cube={dropped}", flush=True)
        if dropped is None or dropped_z >= cz + 0.02 or check_pick_success(dropped):
            print("    drop-carry did not leave a table drop — skip", flush=True)
            return False
        if recorder is not None and record_recovery_only:
            recorder.begin_episode(
                {
                    **episode_meta,
                    "targeted_recovery": True,
                    "recovery_style": "drop_carry",
                    "drop_cube": [float(v) for v in dropped],
                }
            )
            _frame_hook = recorder.on_tick
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        _prepare_regrasp(robot, np.array([cx, cy], dtype=float), cube_yaw_deg)
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        spawn = (cx, cy, cz)
        attempt = _attempt_lift(
            robot,
            recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=0.0,
            close_dy=0.0,
            close_droll=0.0,
            gripper_closed=None,
            phase_prefix="regrasp-",
            contact_reach_m=0.012,
        )
    elif force_first_miss and recovery_style == "closed_empty":
        _attempt_lift(
            robot,
            None,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=0.0,
            close_dy=0.0,
            close_droll=0.0,
            gripper_closed=GRIPPER_OPEN,
            stop_before_close=True,
        )
        parked = _site_pos(robot)
        # The old 8–16 mm miss still closed on the cube. Push the jaws
        # 40 mm off the face so this grasp cannot lift.
        miss_norm = float(np.hypot(miss_dx, miss_dy))
        if miss_norm < 1e-6:
            miss_dx, miss_dy = 0.04, 0.0
        else:
            miss_dx *= 0.040 / miss_norm
            miss_dy *= 0.040 / miss_norm
        servo_xyz(
            robot,
            np.array([parked[0] + miss_dx, parked[1] + miss_dy, GRASP_Z]),
            gripper=GRIPPER_OPEN,
            steps=80,
            phase="closed-empty-offset",
            wrist_target=np.array([WRIST_GRASP[0], _q_arm(robot)[4] + miss_droll]),
            xy_tol=0.004,
            prevent_overshoot=False,
            max_step=0.02,
            kp=1.4,
            dq_lim=0.3,
        )
        moved = float(np.linalg.norm(_site_pos(robot)[:2] - parked[:2]))
        print(f"    closed-empty offset moved {moved * 1000:.1f} mm", flush=True)
        if moved < 0.008:
            print("    offset did not leave the cube — skip", flush=True)
            return False
        set_grip(robot, GRIPPER_CLOSED, steps=GRIPPER_SQUEEZE_STEPS)
        hold(robot, GRIPPER_CLOSED, steps=GRIPPER_HOLD_STEPS)
        lift_xy = _site_pos(robot)
        servo_xyz(
            robot,
            np.array([lift_xy[0], lift_xy[1], LIFT_Z]),
            gripper=GRIPPER_CLOSED,
            steps=60,
            phase="closed-empty-lift",
            lock_wrist=True,
            wrist_target=_q_arm(robot)[3:5].copy(),
            prevent_overshoot=False,
            max_step=1.0,
        )
        lifted_now = (robot.get_block_position() or (0.0, 0.0, cz))[2] >= cz + 0.02
        print(f"    closed-empty lifted={lifted_now}", flush=True)
        if lifted_now:
            print("    fully closed grasp still lifted — skip; no bad labels saved", flush=True)
            return False
        face_normal = _face_standoff(np.array([cx, cy]), cube_yaw_deg, _site_pos(robot)[:2])[1]
        attempt = {
            "lifted": False,
            "rejected": False,
            "wrist_target": _q_arm(robot)[3:5].copy(),
            "face_normal": face_normal,
            "z_after_close": cz,
            "grip_cmd": GRIPPER_CLOSED,
        }
        if recorder is not None and record_recovery_only:
            recorder.begin_episode(
                {
                    **episode_meta,
                    "targeted_recovery": True,
                    "recovery_style": recovery_style,
                    "first_miss_shift_xy_m": [float(miss_dx), float(miss_dy)],
                    "first_miss_roll_rad": float(miss_droll),
                    "closed_empty_offset_m": moved,
                    **miss_meta,
                }
            )
            _frame_hook = recorder.on_tick
    elif force_first_miss:
        attempt = _attempt_lift(
            robot,
            None if record_recovery_only else recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            close_dx=float(miss_dx),
            close_dy=float(miss_dy),
            close_droll=float(miss_droll),
            # Keep the fingers too open to retain the cube.  Together with
            # the small pose error this produces the same useful state as a
            # policy's close-without-lift failure, without a large arm swing.
            gripper_closed=miss_gripper,
        )
        if attempt["lifted"] and record_recovery_only:
            print("    perturbed first grasp unexpectedly lifted — skip; no bad labels saved", flush=True)
            return False
        if recorder is not None and record_recovery_only:
            recorder.begin_episode(
                {
                    **episode_meta,
                    "targeted_recovery": True,
                    "recovery_style": recovery_style,
                    "first_miss_shift_xy_m": [float(miss_dx), float(miss_dy)],
                    "first_miss_roll_rad": float(miss_droll),
                    **miss_meta,
                }
            )
            _frame_hook = recorder.on_tick
    else:
        attempt = _attempt_lift(
            robot,
            recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            contact_reach_m=FIRST_CONTACT_REACH_M,
            joint_kick=joint_kick,
            **grasp_kwargs,
        )
    grasp_attempts = 1
    if allow_regrasp and not attempt["lifted"]:
        print("    first lift failed — open, retreat, regrasp once", flush=True)
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        _prepare_regrasp(robot, np.array([cx, cy], dtype=float), cube_yaw_deg)
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        spawn = (cx, cy, cz)
        attempt = _attempt_lift(
            robot,
            recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            phase_prefix="regrasp-",
            contact_reach_m=0.012,
            **grasp_kwargs,
        )
        grasp_attempts = 2
    elif record_recovery_only and force_first_miss and not attempt["lifted"]:
        # Home-start recovery without the high standoff retreat: open in place,
        # then a normal face approach / close from the current cube pose.
        # An empty close leaves the gripper at lift height, so retreat first.
        if recovery_style == "closed_empty":
            print("    empty close — retreat and regrasp", flush=True)
            cx, cy, cz, cube_yaw_deg = _block_pose(robot)
            _prepare_regrasp(robot, np.array([cx, cy], dtype=float), cube_yaw_deg)
        else:
            print("    first lift failed — open in place, local reclose", flush=True)
            set_grip(robot, GRIPPER_OPEN, steps=GRIPPER_SQUEEZE_STEPS)
            hold(robot, GRIPPER_OPEN, steps=GRIPPER_HOLD_STEPS)
        cx, cy, cz, cube_yaw_deg = _block_pose(robot)
        spawn = (cx, cy, cz)
        attempt = _attempt_lift(
            robot,
            recorder,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw_deg,
            spawn=spawn,
            phase_prefix="correct-",
            contact_reach_m=0.012,
            **grasp_kwargs,
        )
        grasp_attempts = 2
    lifted = bool(attempt["lifted"])
    wrist_target = attempt["wrist_target"]
    z_after_close = float(attempt["z_after_close"])
    grip_cmd = float(attempt["grip_cmd"])
    if recorder is not None:
        recorder.update_metadata(
            lifted=lifted,
            z_after_close_m=z_after_close,
            grasp_attempts=grasp_attempts,
            regrasp=grasp_attempts > 1,
        )
    if attempt["rejected"] and not allow_regrasp:
        if recorder is not None:
            recorder.update_metadata(quality_reject="pre_close")
            recorder.end_episode(success=False, episode=episode)
        return False
    if stop_after_lift:
        block = robot.get_block_position()
        print(
            f"  probe lifted={lifted} grip={grip_cmd:.3f} z_close={z_after_close:.3f} final={block}",
            flush=True,
        )
        return lifted
    if lifted and carry_kick is not None and float(np.linalg.norm(carry_kick)) > 0.0:
        applied = _apply_carry_kick(robot, np.asarray(carry_kick, dtype=float), grip_cmd)
        if recorder is not None:
            recorder.update_metadata(carry_kick=[float(v) for v in applied])
        held = robot.get_block_position()
        still_up = held is not None and float(held[2]) >= cz + 0.015
        if not still_up:
            print(f"    carry-kick dropped the cube cube={held} — discard", flush=True)
            if recorder is not None:
                recorder.end_episode(success=False, episode=episode)
            return False
    if not lifted:
        print("    WARN: cube not in hand after lift — skip place", flush=True)
        block = robot.get_block_position()
        success = check_pick_success(block)
        print(
            f"  result success={success} lifted={lifted} z_close={z_after_close:.3f} "
            f"final={block} grasps={grasp_attempts}",
            flush=True,
        )
        if recorder is not None:
            recorder.end_episode(success=success, episode=episode)
        return success

    ok = True
    # 4) Place in the box.
    ok = (
        servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
            gripper=GRIPPER_CLOSED,
            steps=140,
            phase="to-box",
            lock_wrist=True,
            wrist_target=wrist_target,
            kp=1.6,
            dq_lim=0.35,
            prevent_overshoot=False,
            max_step=1.0,
        )
        == "reached"
        or ok
    )
    if release_kick is not None and float(np.linalg.norm(release_kick)) > 0.0:
        applied = _apply_carry_kick(robot, np.asarray(release_kick, dtype=float), GRIPPER_CLOSED)
        if recorder is not None:
            recorder.update_metadata(release_kick=[float(v) for v in applied])
        held = robot.get_block_position()
        still_up = held is not None and float(held[2]) >= cz + 0.015
        if not still_up:
            print(f"    release-kick dropped the cube cube={held} — discard", flush=True)
            if recorder is not None:
                recorder.end_episode(success=False, episode=episode)
            return False
    hold(robot, GRIPPER_CLOSED, steps=10)
    ok = (
        servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], PLACE_Z]),
            gripper=GRIPPER_CLOSED,
            steps=70,
            phase="lower",
            lock_wrist=True,
            wrist_target=wrist_target,
            prevent_overshoot=False,
            max_step=1.0,
        )
        == "reached"
        or ok
    )
    set_grip(robot, GRIPPER_OPEN, steps=30)
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
        gripper=GRIPPER_OPEN,
        steps=45,
        phase="retreat",
        lock_wrist=False,
        prevent_overshoot=False,
        max_step=1.0,
    )

    block = robot.get_block_position()
    success = check_pick_success(block)
    print(
        f"  result success={success} lifted={lifted} z_close={z_after_close:.3f} final={block} ok_servo={ok}",
        flush=True,
    )
    if recorder is not None:
        recorder.end_episode(success=success, episode=episode)
    return success


def recover_and_place(
    robot: SO101MujocoRobot,
    recorder: ScriptedRecorder | None,
    episode: int,
    episode_meta: dict[str, Any],
    *,
    continue_episode: bool = False,
) -> bool:
    """Expert correction from an already failed grasp.

    When ``continue_episode`` is False (default), starts a new recorder episode
    that contains only the recovery segment. When True, appends into the
    already-open episode (home → policy failure → expert) without resetting
    the frame counter.
    """
    global _frame_hook
    if recorder is not None:
        if continue_episode:
            recorder.update_metadata(**episode_meta)
        else:
            recorder.begin_episode(episode_meta)
        _frame_hook = recorder.on_tick
    cx, cy, cz, cube_yaw_deg = _block_pose(robot)
    _prepare_regrasp(robot, np.array([cx, cy], dtype=float), cube_yaw_deg)
    cx, cy, cz, cube_yaw_deg = _block_pose(robot)
    attempt = _attempt_lift(
        robot,
        recorder,
        cx=cx,
        cy=cy,
        cz=cz,
        cube_yaw_deg=cube_yaw_deg,
        spawn=(cx, cy, cz),
        close_dx=0.0,
        close_dy=0.0,
        close_droll=0.0,
        gripper_closed=None,
        phase_prefix="policy-",
        contact_reach_m=0.012,
        # Real ACT misses often nudge the cube ~11 mm before the expert can close.
        # Keep the 8 mm gate for scripted demos; this path only rejects larger shoves.
        slip_max_m=0.012,
    )
    lifted = bool(attempt["lifted"])
    wrist_target = attempt["wrist_target"]
    if recorder is not None:
        recorder.update_metadata(
            lifted=lifted,
            z_after_close_m=float(attempt["z_after_close"]),
            grasp_attempts=2,
            regrasp=True,
        )
    if not lifted:
        block = robot.get_block_position()
        success = check_pick_success(block)
        print(f"  policy recovery failed final={block}", flush=True)
        if recorder is not None:
            recorder.end_episode(success=success, episode=episode)
        _frame_hook = None
        return success
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
        gripper=GRIPPER_CLOSED,
        steps=140,
        phase="to-box",
        lock_wrist=True,
        wrist_target=wrist_target,
        kp=1.6,
        dq_lim=0.35,
        prevent_overshoot=False,
        max_step=1.0,
    )
    hold(robot, GRIPPER_CLOSED, steps=10)
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], PLACE_Z]),
        gripper=GRIPPER_CLOSED,
        steps=70,
        phase="lower",
        lock_wrist=True,
        wrist_target=wrist_target,
        prevent_overshoot=False,
        max_step=1.0,
    )
    set_grip(robot, GRIPPER_OPEN, steps=30)
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
        gripper=GRIPPER_OPEN,
        steps=45,
        phase="retreat",
        lock_wrist=False,
        prevent_overshoot=False,
        max_step=1.0,
    )
    block = robot.get_block_position()
    success = check_pick_success(block)
    print(f"  policy recovery success={success} final={block}", flush=True)
    if recorder is not None:
        recorder.end_episode(success=success, episode=episode)
    _frame_hook = None
    return success


def build_robot(*, headless: bool) -> SO101MujocoRobot:
    return SO101MujocoRobot(
        SO101MujocoConfig(
            xml_path=Path("SO101/scenes/simple_pick/scene.xml"),
            cube_positions_path=Path("labs/lab01_pnp/configs/cube_positions_demo_fixed.json"),
            record_fps=20,
            control_fps=180,
            physics_fps=360,
            render_window=not headless,
            action_mode="position",
            ee_site_name="gripperframe",
            reset_mode="auto",
            reset_arm="home",
            reset_cube="random",
            cube_random_x_range=[0.26, 0.34],
            cube_random_y_range=[0.165, 0.235],
            cube_random_z=0.0125,
            cube_random_yaw_range=[-45.0, 45.0],
            home_gripper=GRIPPER_OPEN,
            render_frames_per_action=3,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("LAB01_YAW_EPISODES", "3")))
    parser.add_argument(
        "--target-successes",
        type=int,
        default=int(os.environ.get("LAB01_YAW_TARGET_SUCCESSES", "0")),
        help=(
            "Fill balanced yaw-bin and 2x2 XY-cell success quotas. "
            "Resumes counts from spawns.jsonl. 0 means run --episodes attempts."
        ),
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=int(os.environ.get("LAB01_YAW_MAX_ATTEMPTS", "500")),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--v2",
        action="store_true",
        help=(
            "Collect agent-yaw-v2: home jitter, one regrasp, new dataset. "
            "Does not write the original agent-yaw set."
        ),
    )
    parser.add_argument(
        "--target-recovery-successes",
        type=int,
        default=int(os.environ.get("LAB01_YAW_TARGET_RECOVERY_SUCCESSES", "0")),
        help=(
            "Append recovery demonstrations until this many successes of "
            "--recovery-style exist. Without a style, counts every targeted recovery."
        ),
    )
    parser.add_argument(
        "--recovery-style",
        choices=("closed-empty", "pre-close", "drop-carry"),
        default="",
        help="closed-empty: fully closed miss then regrasp. pre-close: correct while still open.",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        default=os.environ.get("LAB01_YAW_RECORD", "") in {"1", "true"},
        help="Write successful episodes to a local LeRobot v3 dataset",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            os.environ.get(
                "LAB01_YAW_DATASET_ROOT",
                "datasets/so101-simstudio-lab01-pnp-agent-yaw",
            )
        ),
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get(
            "LAB01_YAW_REPO_ID",
            "alexhegit/so101-simstudio-lab01-pnp-agent-yaw",
        ),
    )
    parser.add_argument(
        "--joint-diverse",
        action="store_true",
        help=(
            "After the face standoff, kick shoulder / elbow / wrist_flex off the "
            "nominal tube, then let the expert correct. Keep only full successes. "
            "Does not overwrite an existing dataset; resume appends."
        ),
    )
    parser.add_argument(
        "--carry-diverse",
        action="store_true",
        help=(
            "After a successful lift, kick shoulder / elbow / wrist_flex while the "
            "gripper stays closed, then place and release. Keep only full successes."
        ),
    )
    parser.add_argument(
        "--release-diverse",
        action="store_true",
        help=(
            "After the arm reaches the container, kick shoulder / elbow / wrist_flex "
            "with the gripper closed, then lower and open. Keep only full successes."
        ),
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        default=os.environ.get("LAB01_YAW_SAVE_FAILURES", "") in {"1", "true"},
    )
    args = parser.parse_args()
    if args.v2:
        root = Path(args.dataset_root)
        if "agent-yaw-v2" not in root.name:
            args.dataset_root = root.with_name("so101-simstudio-lab01-pnp-agent-yaw-v2")
        if not str(args.repo_id).endswith("agent-yaw-v2"):
            args.repo_id = "alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v2"
        args.record = True
        if args.target_successes <= 0 and args.target_recovery_successes <= 0:
            args.target_successes = 80
        print(
            f"v2 collect: jitter+regrasp → {args.dataset_root} target={args.target_successes}",
            flush=True,
        )
    headless = args.headless or os.environ.get("LAB01_YAW_HEADLESS", "") in {"1", "true"}
    global STEP_SLEEP_S, _frame_hook
    if headless:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.pop("DISPLAY", None)
        STEP_SLEEP_S = 0.0
    else:
        if not os.environ.get("DISPLAY"):
            print("DISPLAY is unset. On .43 run from a graphical session:", file=sys.stderr)
            print("  ./labs/lab01_pnp/auto_collect_yaw.cmd", file=sys.stderr)
            return 2
        os.environ.setdefault("MUJOCO_GL", "glfw")

    robot = build_robot(headless=headless)
    robot.connect()
    recorder: ScriptedRecorder | None = None
    if args.record:
        recorder = ScriptedRecorder(
            robot,
            repo_id=args.repo_id,
            root=args.dataset_root,
            save_failures=args.save_failures,
        )
        _frame_hook = recorder.on_tick
        print(f"recording → {args.dataset_root} ({args.repo_id})", flush=True)

    wins = 0
    recovery_wins = 0
    diverse_wins = 0
    carry_wins = 0
    release_wins = 0
    style_name = {
        "closed-empty": "closed_empty",
        "pre-close": "pre_close",
        "drop-carry": "drop_carry",
    }.get(args.recovery_style, "")
    bin_wins = [0] * 6
    xy_wins = [0] * len(XY_CELLS)
    if recorder is not None and recorder.spawn_log.is_file():
        for line in recorder.spawn_log.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("success"):
                continue
            if style_name:
                if row.get("recovery_style") == style_name:
                    recovery_wins += 1
            elif row.get("targeted_recovery"):
                recovery_wins += 1
            if row.get("joint_kick"):
                diverse_wins += 1
            if row.get("carry_kick"):
                carry_wins += 1
            if row.get("release_kick"):
                release_wins += 1
            if row.get("yaw_bin") is not None:
                bin_wins[int(row["yaw_bin"])] += 1
            cell = row.get("xy_cell")
            if cell in XY_CELL_NAMES:
                xy_wins[XY_CELL_NAMES.index(cell)] += 1
        wins = sum(bin_wins)

    if (args.joint_diverse or args.carry_diverse or args.release_diverse) and args.target_successes <= 0:
        args.target_successes = 50 if args.joint_diverse else 24
    target = max(0, args.target_successes)
    recovery_target = max(0, args.target_recovery_successes)
    quotas = split_quota(target, 6, YAW_REMAINDER_ORDER) if target else [0] * 6
    # low-Y cells first in the remainder so a 50→100 top-up prefers the empty band.
    xy_quotas = split_quota(target, 4, XY_REMAINDER_ORDER) if target else [0] * 4
    if target:
        print(f"yaw-bin target={quotas} existing={bin_wins}", flush=True)
        print(f"xy-cell target={xy_quotas} existing={xy_wins} names={XY_CELL_NAMES}", flush=True)
    if recovery_target:
        print(f"targeted recovery target={recovery_target} existing={recovery_wins}", flush=True)
    if args.joint_diverse:
        print(
            f"joint-diverse target={target} existing_kicks={diverse_wins} "
            f"patterns={len(JOINT_KICKS)}",
            flush=True,
        )
    if args.carry_diverse:
        print(
            f"carry-diverse target={target} existing_carry={carry_wins} "
            f"patterns={len(CARRY_KICKS)}",
            flush=True,
        )
    if args.release_diverse:
        print(
            f"release-diverse target={target} existing_release={release_wins} "
            f"patterns={len(RELEASE_KICKS)}",
            flush=True,
        )

    attempts = 0
    bin_cursor = 0
    xy_cursor = 0
    try:
        while attempts < (
            args.max_attempts if (target or recovery_target or args.v2) else args.episodes
        ):
            if recovery_target and recovery_wins >= recovery_target:
                break
            if args.joint_diverse and diverse_wins >= target:
                break
            if args.carry_diverse and carry_wins >= target:
                break
            if args.release_diverse and release_wins >= target:
                break
            if args.v2 and wins >= target:
                if not recovery_target:
                    break
            if (
                target
                and not recovery_target
                and not args.v2
                and not args.joint_diverse
                and not args.carry_diverse
                and not args.release_diverse
                and all(bin_wins[i] >= quotas[i] for i in range(6))
                and all(xy_wins[i] >= xy_quotas[i] for i in range(4))
            ):
                break
            if (
                target
                and not recovery_target
                and not args.v2
                and not args.joint_diverse
                and not args.carry_diverse
                and not args.release_diverse
            ):
                yaw_need = next_deficit(bin_wins, quotas, bin_cursor)
                xy_need = next_deficit(xy_wins, xy_quotas, xy_cursor)
                yaw_bin = yaw_need if yaw_need is not None else (bin_cursor % 6)
                xy_idx = xy_need if xy_need is not None else (xy_cursor % 4)
                bin_cursor = yaw_bin + 1
                xy_cursor = xy_idx + 1
            else:
                yaw_bin = attempts % 6
                xy_idx = attempts % 4
            cell_name, x_range, y_range = XY_CELLS[xy_idx]
            robot.config.cube_random_yaw_range = [
                float(YAW_BIN_EDGES_DEG[yaw_bin]),
                float(YAW_BIN_EDGES_DEG[yaw_bin + 1]),
            ]
            robot.config.cube_random_x_range = list(x_range)
            robot.config.cube_random_y_range = list(y_range)
            kick = None
            carry = None
            if args.joint_diverse:
                kick = np.array(JOINT_KICKS[attempts % len(JOINT_KICKS)], dtype=float)
            if args.carry_diverse:
                carry = np.array(CARRY_KICKS[attempts % len(CARRY_KICKS)], dtype=float)
            release = None
            if args.release_diverse:
                release = np.array(RELEASE_KICKS[attempts % len(RELEASE_KICKS)], dtype=float)
            success = run_episode(
                robot,
                attempts,
                random_cube=robot.config.reset_cube == "random",
                recorder=recorder,
                start_jitter_rad=0.06 if args.v2 else 0.0,
                joint_kick=kick,
                carry_kick=carry,
                release_kick=release,
                drop_after_lift=args.recovery_style == "drop-carry",
                allow_regrasp=args.v2 and args.recovery_style != "pre-close",
                force_first_miss=recovery_target > 0 and args.recovery_style != "pre-close",
                record_recovery_only=recovery_target > 0,
                recovery_style=style_name or "diverse",
                miss_gripper=GRIPPER_CLOSED if args.recovery_style == "closed-empty" else TARGETED_MISS_GRIPPER,
                miss_large_only=args.recovery_style == "closed-empty",
                preclose=args.recovery_style == "pre-close",
            )
            if success:
                wins += 1
                bin_wins[yaw_bin] += 1
                xy_wins[xy_idx] += 1
                meta = recorder._attempt_meta if recorder is not None else {}
                if meta.get("joint_kick"):
                    diverse_wins += 1
                if meta.get("carry_kick"):
                    carry_wins += 1
                if meta.get("release_kick"):
                    release_wins += 1
                if style_name:
                    if meta.get("recovery_style") == style_name:
                        recovery_wins += 1
                elif meta.get("targeted_recovery") or meta.get("regrasp"):
                    recovery_wins += 1
            attempts += 1
            print(
                f"  campaign attempts={attempts} wins={wins} bins={bin_wins} "
                f"xy={xy_wins} recovery={recovery_wins} diverse={diverse_wins} "
                f"carry={carry_wins} release={release_wins} cell={cell_name}"
                + (f" target_yaw={quotas} target_xy={xy_quotas}" if target else ""),
                flush=True,
            )
            if not headless:
                time.sleep(0.4)
    finally:
        _frame_hook = None
        if recorder is not None:
            recorder.close()
        robot.disconnect()
    print(f"yaw-aware scripted expert: wins={wins} attempts={attempts} bins={bin_wins} xy={xy_wins}")
    if recovery_target:
        return 0 if recovery_wins >= recovery_target else 1
    if args.joint_diverse:
        return 0 if diverse_wins >= target else 1
    if args.carry_diverse:
        return 0 if carry_wins >= target else 1
    if args.release_diverse:
        return 0 if release_wins >= target else 1
    if args.v2:
        return 0 if wins >= target else 1
    if target:
        yaw_ok = all(bin_wins[i] >= quotas[i] for i in range(6))
        xy_ok = all(xy_wins[i] >= xy_quotas[i] for i in range(4))
        return 0 if yaw_ok and xy_ok else 1
    return 0 if wins > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
