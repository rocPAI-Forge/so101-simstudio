#!/usr/bin/env python3
"""Visual / record PoC: scripted grasp (no LLM, no ACT) on Lab 01 simple_pick.

Joint-space path toward Q_READY decelerates by distance to the cube and
settles in place (no hard brake / snap-back), then creeps and closes.

Visual only:
    ./labs/lab01_pnp/auto_collect_poc.cmd

Record successful episodes to a local LeRobot v3 dataset (position actions):
    LAB01_POC_RECORD=1 LAB01_POC_EPISODES=5 ./labs/lab01_pnp/auto_collect_poc.cmd
    LAB01_POC_RECORD=1 LAB01_POC_HEADLESS=1 MUJOCO_GL=osmesa \\
      LAB01_POC_EPISODES=10 ./labs/lab01_pnp/auto_collect_poc.cmd
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

# Import LeRobot *before* mujoco — otherwise OSMesa+torch import order can SIGSEGV.
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

import mujoco as mj
import numpy as np

from simstudio.common.eval_success import CONTAINER_XY, check_pick_success
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
DT = 0.05
STEP_SLEEP_S = float(os.environ.get("LAB01_POC_SLEEP", "0.03"))
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
            self.dataset = LeRobotDataset(
                repo_id,
                root=self.root,
                tolerance_s=1e-3,
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
        self._spawn: tuple[float, float, float] | None = None
        self.saved = 0
        self.skipped = 0

    def begin_episode(self, spawn: tuple[float, float, float]) -> None:
        self._spawn = spawn
        self._ep_frames = 0

    def on_tick(self, obs: dict[str, Any], action: dict[str, float]) -> None:
        obs_frame = build_dataset_frame(self.features, obs, prefix=OBS_STR)
        act_frame = build_dataset_frame(self.features, action, prefix=ACTION)
        self.dataset.add_frame({**obs_frame, **act_frame, "task": self.task})
        self._ep_frames += 1

    def end_episode(self, *, success: bool, episode: int) -> None:
        meta = {
            "episode": episode,
            "success": success,
            "frames": self._ep_frames,
            "spawn": list(self._spawn) if self._spawn is not None else None,
        }
        with self.spawn_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(meta) + "\n")
        if success or self.save_failures:
            if self._ep_frames > 0:
                self.dataset.save_episode()
                self.saved += 1
                print(
                    f"  saved episode frames={self._ep_frames} success={success} "
                    f"total_saved={self.saved}",
                    flush=True,
                )
            else:
                self.skipped += 1
        else:
            self.dataset.clear_episode_buffer()
            self.skipped += 1
            print(
                f"  discard failed episode frames={self._ep_frames} "
                f"(set --save-failures to keep)",
                flush=True,
            )

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
            why = (
                "soft-stop"
                if close_enough
                else ("passed +X" if passed else "receding")
            )
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
    xy_tol: float = XY_TOL,
    z_tol: float = Z_TOL,
    kp: float = KP,
    dq_lim: float = DQ_LIM,
    max_step: float = 0.012,
    prevent_overshoot: bool = True,
    stop_on_cube_move: bool = False,
    cube_ref: tuple[float, float, float] | None = None,
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
    cols = [robot.dof_ids[n] for n in ARM[:n_arm]]
    sid = _site_id(robot)
    q = _q_arm(robot)
    if lock_wrist:
        q[3:] = WRIST_GRASP
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
                if moved >= CONTACT_MOVE_M:
                    for _hold in range(8):
                        _send_q(robot, q, gripper)
                    moved = float(
                        np.linalg.norm(
                            np.asarray(robot.get_block_position()[:2])
                            - np.asarray(cube_ref[:2])
                        )
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
            q_trial[3:] = WRIST_GRASP

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


def run_episode(
    robot: SO101MujocoRobot,
    episode: int,
    *,
    random_cube: bool,
    recorder: ScriptedRecorder | None = None,
) -> bool:
    robot.reset_episode(episode)
    cube = robot.get_block_position()
    if cube is None:
        print("no cube", file=sys.stderr)
        return False
    cx, cy, cz = cube
    print(f"episode {episode} spawn=({cx:.3f}, {cy:.3f}, {cz:.3f}) random={random_cube}", flush=True)

    spawn = (cx, cy, cz)
    if recorder is not None:
        recorder.begin_episode(spawn)
    grasp = np.array([cx, cy, GRASP_Z])
    cube_xy = np.array([cx, cy], dtype=float)

    # 1) Move toward elbow-up ready but DO NOT park past the cube.
    #    Full goto_q(Q_READY) swings home → near cube → far side; the old
    #    slow "approach" was just walking back from that overshoot.
    goto_ready_without_passing_cube(
        robot, cube_xy, gripper=GRIPPER_OPEN, steps=50
    )
    # Snap wrist to grasp posture; hold XY with a few servo steps if it drifts.
    q = _q_arm(robot)
    q[3:] = WRIST_GRASP
    for _ in range(3):
        _send_q(robot, q, GRIPPER_OPEN)
    print(f"    after-ready gf={np.round(_site_pos(robot), 4)}", flush=True)

    # 2) Short creep into soft wrap (already beside the cube — no long return).
    status = servo_xyz(
        robot,
        grasp,
        gripper=GRIPPER_OPEN,
        steps=70,
        phase="creep",
        xy_tol=XY_TOL_CREEP,
        kp=1.7,
        dq_lim=0.28,
        max_step=0.015,
        prevent_overshoot=False,
    )
    ok = status == "reached"
    print(
        f"    after-creep status={status} gf={np.round(_site_pos(robot), 4)} "
        f"cube={robot.get_block_position()}",
        flush=True,
    )

    gf = _site_pos(robot)
    cube_now = robot.get_block_position()
    xy_err = float(np.linalg.norm(gf[:2] - np.asarray(spawn[:2])))
    cube_slip = (
        float(np.linalg.norm(np.asarray(cube_now[:2]) - np.asarray(spawn[:2])))
        if cube_now is not None
        else 0.0
    )
    print(
        f"    pre-close status={status} gf={np.round(gf, 4)} cube={cube_now} "
        f"xy_to_spawn={xy_err:.4f} cube_slip={cube_slip * 1000:.1f}mm",
        flush=True,
    )

    # 3) Close with arm frozen — no further XY chase.
    set_grip(robot, GRIPPER_CLOSED, steps=GRIPPER_SQUEEZE_STEPS)
    hold(robot, GRIPPER_CLOSED, steps=GRIPPER_HOLD_STEPS)
    z_after_close = (robot.get_block_position() or (0.0, 0.0, 0.0))[2]

    hold_xy = _site_pos(robot)
    ok = (
        servo_xyz(
            robot,
            np.array([hold_xy[0], hold_xy[1], LIFT_Z]),
            gripper=GRIPPER_CLOSED,
            steps=100,
            phase="lift",
            lock_wrist=False,
            prevent_overshoot=False,
            max_step=1.0,
        )
        == "reached"
        or ok
    )
    lifted = (robot.get_block_position() or (0.0, 0.0, 0.0))[2] >= cz + 0.02
    print(f"    after-lift cube={robot.get_block_position()} lifted={lifted}", flush=True)
    if not lifted:
        print("    WARN: cube not in hand after lift — skip place", flush=True)
        block = robot.get_block_position()
        success = check_pick_success(block)
        print(
            f"  result success={success} lifted={lifted} z_close={z_after_close:.3f} "
            f"final={block} ok_servo={ok}",
            flush=True,
        )
        if recorder is not None:
            recorder.end_episode(success=success, episode=episode)
        return success

    # 4) Place in the box.
    ok = (
        servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
            gripper=GRIPPER_CLOSED,
            steps=140,
            phase="to-box",
            lock_wrist=False,
            kp=1.6,
            dq_lim=0.35,
            prevent_overshoot=False,
            max_step=1.0,
        )
        == "reached"
        or ok
    )
    hold(robot, GRIPPER_CLOSED, steps=10)
    ok = (
        servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], PLACE_Z]),
            gripper=GRIPPER_CLOSED,
            steps=70,
            phase="lower",
            lock_wrist=False,
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
        f"  result success={success} lifted={lifted} z_close={z_after_close:.3f} "
        f"final={block} ok_servo={ok}",
        flush=True,
    )
    if recorder is not None:
        recorder.end_episode(success=success, episode=episode)
    return success


def build_robot(*, headless: bool) -> SO101MujocoRobot:
    random_cube = os.environ.get("LAB01_POC_RANDOM", "") in {"1", "true"}
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
            reset_cube="random" if random_cube else "fixed",
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
    parser.add_argument("--episodes", type=int, default=int(os.environ.get("LAB01_POC_EPISODES", "3")))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--record",
        action="store_true",
        default=os.environ.get("LAB01_POC_RECORD", "") in {"1", "true"},
        help="Write successful episodes to a local LeRobot v3 dataset",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("LAB01_POC_DATASET_ROOT", ".tmp/lab01_scripted_poc")),
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("LAB01_POC_REPO_ID", "local/lab01-scripted-poc"),
    )
    parser.add_argument(
        "--save-failures",
        action="store_true",
        default=os.environ.get("LAB01_POC_SAVE_FAILURES", "") in {"1", "true"},
    )
    args = parser.parse_args()
    headless = args.headless or os.environ.get("LAB01_POC_HEADLESS", "") in {"1", "true"}
    global STEP_SLEEP_S, _frame_hook
    if headless:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        os.environ.pop("DISPLAY", None)
        STEP_SLEEP_S = 0.0
    else:
        if not os.environ.get("DISPLAY"):
            print("DISPLAY is unset. On .43 run from a graphical session:", file=sys.stderr)
            print("  ./labs/lab01_pnp/auto_collect_poc.cmd", file=sys.stderr)
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
    try:
        for ep in range(args.episodes):
            if run_episode(
                robot,
                ep,
                random_cube=robot.config.reset_cube == "random",
                recorder=recorder,
            ):
                wins += 1
            time.sleep(0.4)
    finally:
        _frame_hook = None
        if recorder is not None:
            recorder.close()
        robot.disconnect()
    print(f"scripted PoC: {wins}/{args.episodes} placed in box")
    return 0 if wins > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
