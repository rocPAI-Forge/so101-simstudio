#!/usr/bin/env python3
"""Let the current policy run, then record an expert correction from the failure.

Triggers follow the fresh 50-episode failures of the 17/20 checkpoint:

- ``table``: the cube is still on the table well into the episode, or it has
  been pushed outside the spawn box and away from the container. The expert
  regrasps and places. A cube already next to the container is left alone
  until the episode ends, so a slow place is not interrupted.
- ``held``: the cube is lifted and either stalled away from the container or
  held over the container without being placed. The expert finishes the place.

Only the expert segment is recorded, and only full placements are kept.
Policy actions before the handoff are not written.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from auto_collect_yaw import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    LIFT_Z,
    PLACE_Z,
    ScriptedRecorder,
    _q_arm,
    recover_and_place,
    servo_xyz,
    set_grip,
)
from simstudio.common.eval_success import CONTAINER_XY, check_pick_success, cube_over_container

# Spawn box in rollout_smolvla.yaml, expanded by about 1 cm.
_X_LO, _X_HI = 0.25, 0.35
_Y_LO, _Y_HI = 0.15, 0.25
_TABLE_Z = 0.04
_LIFT_Z = 0.06
_CARRY_FAR_M = 0.10
_CARRY_HOLD_TICKS = 100
_CARRY_WINDOW = 60
_HELD_OVER_TICKS = 80
_TABLE_STUCK_TICK = 800
# Container center is outside the spawn box. A cube that has been carried
# there must not count as "pushed away".
_BOX_MARGIN_M = 0.09


def _near_container(x: float, y: float) -> bool:
    return abs(x - CONTAINER_XY[0]) <= _BOX_MARGIN_M and abs(y - CONTAINER_XY[1]) <= _BOX_MARGIN_M


class _Watch:
    def __init__(self) -> None:
        self.lift_ticks = 0
        self.over_ticks = 0
        self.dists: list[float] = []

    def update(self, robot, tick: int) -> str | None:
        block = robot.get_block_position()
        if block is None or check_pick_success(block):
            self.lift_ticks = 0
            self.over_ticks = 0
            self.dists.clear()
            return None
        x, y, z = (float(v) for v in block)
        outside = x < _X_LO or x > _X_HI or y < _Y_LO or y > _Y_HI
        away = not _near_container(x, y)
        pushed = outside and away and tick >= 40
        stuck = away and tick >= _TABLE_STUCK_TICK
        if z < _TABLE_Z and (pushed or stuck):
            return "table"
        if z >= _LIFT_Z and cube_over_container(block):
            self.over_ticks += 1
            if self.over_ticks >= _HELD_OVER_TICKS:
                return "held"
            return None
        self.over_ticks = 0
        if z >= _LIFT_Z:
            dist = float(np.hypot(x - CONTAINER_XY[0], y - CONTAINER_XY[1]))
            self.lift_ticks += 1
            self.dists.append(dist)
            window = self.dists[-_CARRY_WINDOW:]
            stalled = len(window) >= _CARRY_WINDOW and window[-1] > window[0] - 0.02
            if self.lift_ticks >= _CARRY_HOLD_TICKS and dist > _CARRY_FAR_M and stalled:
                return "held"
            return None
        self.lift_ticks = 0
        self.dists.clear()
        return None


def _place_held(robot, recorder, episode: int, meta: dict) -> bool:
    recorder.begin_episode(meta)
    import auto_collect_yaw as collect

    collect._frame_hook = recorder.on_tick
    wrist = _q_arm(robot)[3:5].copy()
    set_grip(robot, GRIPPER_CLOSED, steps=12)
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
        gripper=GRIPPER_CLOSED,
        steps=140,
        phase="takeover-to-box",
        lock_wrist=True,
        wrist_target=wrist,
        kp=1.6,
        dq_lim=0.35,
        prevent_overshoot=False,
        max_step=1.0,
    )
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], PLACE_Z]),
        gripper=GRIPPER_CLOSED,
        steps=70,
        phase="takeover-lower",
        lock_wrist=True,
        wrist_target=wrist,
        prevent_overshoot=False,
        max_step=1.0,
    )
    set_grip(robot, GRIPPER_OPEN, steps=30)
    servo_xyz(
        robot,
        np.array([CONTAINER_XY[0], CONTAINER_XY[1], LIFT_Z]),
        gripper=GRIPPER_OPEN,
        steps=45,
        phase="takeover-retreat",
        lock_wrist=False,
        prevent_overshoot=False,
        max_step=1.0,
    )
    block = robot.get_block_position()
    success = check_pick_success(block)
    print(f"  carry takeover success={success} final={block}", flush=True)
    recorder.end_episode(success=success, episode=episode)
    collect._frame_hook = None
    return success


def _mode_at_end(robot) -> str | None:
    block = robot.get_block_position()
    if block is None or check_pick_success(block):
        return None
    x, y, z = (float(v) for v in block)
    if z >= _LIFT_Z:
        return "held"
    if z < _TABLE_Z:
        return "table"
    return None


def run_takeover_campaign(strategy, ctx, robot, cfg) -> None:
    root = Path(os.environ.get("LAB01_TAKEOVER_ROOT", "")).resolve()
    if not str(os.environ.get("LAB01_TAKEOVER_ROOT", "")):
        raise SystemExit("LAB01_TAKEOVER_ROOT is required")
    info = root / "meta" / "info.json"
    if root.exists() and any(root.iterdir()) and not info.is_file():
        raise SystemExit(f"LAB01_TAKEOVER_ROOT is not a dataset: {root}")
    target_push = int(os.environ.get("LAB01_TAKEOVER_TABLE", os.environ.get("LAB01_TAKEOVER_PUSH", "12")))
    target_carry = int(os.environ.get("LAB01_TAKEOVER_HELD", os.environ.get("LAB01_TAKEOVER_CARRY", "12")))
    max_attempts = int(os.environ.get("LAB01_TAKEOVER_MAX", "140"))
    repo_id = os.environ.get(
        "LAB01_TAKEOVER_REPO",
        "alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4",
    )
    recorder = ScriptedRecorder(robot, repo_id=repo_id, root=root)
    table_wins = 0
    held_wins = 0
    print(
        f"takeover target table={target_push} held={target_carry} max={max_attempts} "
        f"root={root} policy={getattr(getattr(cfg, 'policy', None), 'pretrained_path', '')}",
        flush=True,
    )
    try:
        for attempt in range(max_attempts):
            if table_wins >= target_push and held_wins >= target_carry:
                break
            if ctx.runtime.shutdown_event.is_set():
                break
            robot.reset_episode(attempt)
            strategy._engine.reset()
            strategy._interpolator.reset()
            spawn = robot.get_block_position()
            watch = _Watch()

            def stop_when(inner, tick: int, watch: _Watch = watch) -> str | None:
                return watch.update(inner, tick)

            from simstudio.scripts.eval import _run_episode

            timing = _run_episode(
                strategy,
                ctx,
                60.0,
                stop_when=stop_when,
            )
            strategy._engine.pause()
            reason = timing.get("stop_reason") or _mode_at_end(robot)
            block = robot.get_block_position()
            print(
                f"attempt {attempt} stop={reason} ticks={int(timing['tick_n'])} "
                f"cube={block} table={table_wins} held={held_wins}",
                flush=True,
            )
            if reason not in {"table", "held"}:
                continue
            if reason == "table" and table_wins >= target_push:
                continue
            if reason == "held" and held_wins >= target_carry:
                continue
            meta = {
                "source": "policy_takeover",
                "takeover": reason,
                "spawn": [float(v) for v in spawn] if spawn is not None else None,
                "cube_at_takeover": [float(v) for v in block] if block is not None else None,
            }
            if reason == "table":
                ok = recover_and_place(robot, recorder, attempt, meta)
                if ok:
                    table_wins += 1
            else:
                ok = _place_held(robot, recorder, attempt, meta)
                if ok:
                    held_wins += 1
    finally:
        recorder.close()
    print(f"takeover done table={table_wins} held={held_wins}", flush=True)
