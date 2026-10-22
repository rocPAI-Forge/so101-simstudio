#!/usr/bin/env python3
"""Record expert corrections from ACT close-without-lift states.

Two modes:

* ``mid`` (legacy): start recording only at handover; policy actions discarded.
* ``full`` (DAgger): record from home through policy failure and expert recovery
  as one continuous successful episode. Style tag: ``policy_dagger_full``.

Example (full DAgger onto v4):

  .venv-rocm/bin/python labs/lab01_pnp/collect_policy_recovery.py \\
    --mode full --target 20 --max-attempts 120 \\
    --policy-path outputs/train/lab01_pnp_act_agent_yaw_v3/checkpoints/080000/pretrained_model \\
    --dataset-root datasets/so101-simstudio-lab01-pnp-agent-yaw-v4 \\
    --dataset-repo-id alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4 \\
    --stats-root datasets/so101-simstudio-lab01-pnp-agent-yaw-v3 \\
    --stats-repo-id alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import yaml

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.pop("DISPLAY", None)

from simstudio.scripts.eval import (  # noqa: E402
    _expand_rollout_state_features,
    _fix_preprocessor_normalizer_stats,
)

_SPEC = importlib.util.spec_from_file_location(
    "auto_collect_yaw",
    Path(__file__).with_name("auto_collect_yaw.py"),
)
yaw = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules["auto_collect_yaw"] = yaw
_SPEC.loader.exec_module(yaw)

STYLE_MID = "policy_handover"
STYLE_FULL = "policy_dagger_full"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=("mid", "full"),
        default="full",
        help="mid=recovery-only (legacy); full=home→fail→expert continuous DAgger.",
    )
    p.add_argument(
        "--policy-path",
        type=Path,
        default=Path("outputs/train/lab01_pnp_act_agent_yaw_v3/checkpoints/080000/pretrained_model"),
    )
    p.add_argument(
        "--rollout-config",
        type=Path,
        default=Path("labs/lab01_pnp_agent/configs/rollout_act_agent_yaw.yaml"),
        help="Must match agent-yaw (gripperframe + home_gripper 0.85).",
    )
    p.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets/so101-simstudio-lab01-pnp-agent-yaw-v4"),
    )
    p.add_argument(
        "--dataset-repo-id",
        default="alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4",
    )
    p.add_argument(
        "--stats-root",
        type=Path,
        default=Path("datasets/so101-simstudio-lab01-pnp-agent-yaw-v3"),
        help="Normalizer stats dataset (usually the policy's train set).",
    )
    p.add_argument(
        "--stats-repo-id",
        default="alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v3",
    )
    p.add_argument("--target", type=int, default=20, help="Successful episodes to keep.")
    p.add_argument("--max-attempts", type=int, default=120)
    p.add_argument(
        "--n-action-steps",
        type=int,
        default=50,
        help="ACT chunk size. Pass 0 to omit (LingBot-VLA-v2 / policies without n_action_steps).",
    )
    p.add_argument(
        "--policy-tag",
        default="lab01_pnp_act_agent_yaw_v3/080000",
        help="Stored in spawn metadata only.",
    )
    p.add_argument(
        "--register-v2",
        action="store_true",
        help="Import LingBot-VLA-v2 config (needs PYTHONPATH to PR #3967 side tree).",
    )
    p.add_argument(
        "--closed-hold-ticks",
        type=int,
        default=40,
        help="Ticks of closed gripper without lift before failed_close (20 Hz → 40=2s).",
    )
    return p.parse_args()


def _style(mode: str) -> str:
    return STYLE_FULL if mode == "full" else STYLE_MID


def _rollout_config(args: argparse.Namespace):
    import draccus
    from lerobot.rollout import RolloutConfig

    raw = yaml.safe_load(args.rollout_config.read_text()) or {}
    raw.pop("eval", None)
    raw.setdefault("strategy", {"type": "base"})
    robot = raw.setdefault("robot", {})
    robot["ee_site_name"] = "gripperframe"
    robot["home_gripper"] = 0.85
    robot["reset_arm"] = "home"
    robot["render_window"] = False
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(raw, tmp, sort_keys=False)
        tmp_path = tmp.name
    cli = [
        f"--policy.path={args.policy_path}",
        "--robot.reset_arm=home",
        "--robot.ee_site_name=gripperframe",
        "--robot.home_gripper=0.85",
        "--robot.render_window=false",
        "--display_data=false",
        "--play_sounds=false",
        "--fps=20",
    ]
    if int(args.n_action_steps) > 0:
        cli.insert(1, f"--policy.n_action_steps={args.n_action_steps}")
    sys.argv = [sys.argv[0], *cli]
    draccus_args = [arg for arg in cli if not arg.startswith("--policy.")]
    return draccus.parse(
        config_class=RolloutConfig,
        config_path=tmp_path,
        args=draccus_args,
    )


def _policy_outcome(
    strategy,
    ctx,
    initial_z: float,
    *,
    recorder: yaw.ScriptedRecorder | None = None,
    closed_hold_ticks: int = 40,
    max_ticks: int = 1200,
) -> tuple[str, int]:
    """Run until close-without-lift, lift, or timeout.

    When ``recorder`` is set, every policy tick is written (full DAgger mode).
    Returns ``(outcome, policy_frames)``.
    """
    from lerobot.rollout.strategies.core import send_next_action
    from lerobot.utils.robot_utils import precise_sleep

    robot = ctx.hardware.robot_wrapper
    inner = getattr(robot, "inner", robot)
    engine = strategy._engine
    interpolator = strategy._interpolator
    get_interval = getattr(interpolator, "get_control_interval", None)
    if get_interval is not None:
        control_interval = get_interval(ctx.runtime.cfg.fps)
    else:
        control_interval = 1.0 / (float(ctx.runtime.cfg.fps) * float(getattr(interpolator, "multiplier", 1) or 1))
    from collections import deque

    seen_open = False
    policy_frames = 0
    # Track *observed* jaw and commanded gripper (V2 chatters both).
    obs_grip_min = float("inf")
    obs_grip_max = float("-inf")
    cmd_grip_min = float("inf")
    cmd_grip_max = float("-inf")
    near_cube = False
    min_dist = float("inf")
    # Rolling window: V2 jaw chatters open/closed; consecutive-hold never fires.
    # Need ~60% closed ticks in the last ``closed_hold_ticks`` (default 40 → 24).
    window = max(int(closed_hold_ticks), 10)
    need = max(int(round(window * 0.6)), 8)
    recent_closed: deque[bool] = deque(maxlen=window)
    # Also accumulate closed-near ticks across the episode (chatter-tolerant).
    near_closed_ticks = 0
    engine.resume()
    try:
        for _tick in range(max_ticks):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = strategy._process_observation_and_notify(ctx.processors, obs)
            sent = send_next_action(obs_processed, obs, ctx, interpolator)
            if recorder is not None and sent is not None:
                recorder.on_tick(obs, sent)
                policy_frames += 1
            if sent is None:
                if (sleep_t := control_interval - (time.perf_counter() - loop_start)) > 0:
                    precise_sleep(sleep_t)
                continue
            # Observed jaw — matches Home20 empty-grasp (jaw≈0) diagnosis.
            if "gripper.pos" in obs:
                gripper = float(obs["gripper.pos"])
            else:
                gripper = float("nan")
            if gripper == gripper:
                obs_grip_min = min(obs_grip_min, gripper)
                obs_grip_max = max(obs_grip_max, gripper)
            if "gripper.pos" in sent:
                cmd = float(sent["gripper.pos"])
                cmd_grip_min = min(cmd_grip_min, cmd)
                cmd_grip_max = max(cmd_grip_max, cmd)
            block = inner.get_block_position()
            lifted = block is not None and block[2] >= initial_z + 0.02
            if block is not None:
                site = inner.data.site_xpos[inner.ee_site_id]
                dist = (
                    (float(site[0]) - block[0]) ** 2
                    + (float(site[1]) - block[1]) ** 2
                    + (float(site[2]) - block[2]) ** 2
                ) ** 0.5
                min_dist = min(min_dist, dist)
                # Soft near: Home20 first-close standoff ~27 mm.
                near_cube = dist <= 0.06
            if gripper >= 0.45:
                seen_open = True
            # Observed jaw. Soft ≤0.35 near cube catches V2 chatter; hard ≤0.15
            # is the Home20 empty-grasp. Expert regrasp now lifts out of the
            # low basin first (auto_collect_yaw._prepare_regrasp).
            hard_closed = seen_open and gripper <= 0.15
            soft_closed = seen_open and near_cube and gripper <= 0.35
            closed = hard_closed or soft_closed
            if closed and lifted:
                print(
                    f"  policy lifted obs_grip={gripper:.3f} min_dist={min_dist:.3f} "
                    f"frames={policy_frames}",
                    flush=True,
                )
                return "lifted", policy_frames
            # Empty-grasp without lift. Rolling majority + cumulative near-cube
            # closed ticks (V2 jaw chatter zeros a consecutive counter).
            is_empty_close = bool(closed and not lifted)
            recent_closed.append(is_empty_close)
            if is_empty_close and near_cube:
                near_closed_ticks += 1
            window_hit = len(recent_closed) >= window and sum(recent_closed) >= need
            # 20 ticks (~1 s) near+closed is enough to hand over; 40 was too
            # strict when the jaw only briefly dips below 0.15.
            cumul_need = max(closed_hold_ticks // 2, 20)
            cumul_hit = near_closed_ticks >= cumul_need
            if window_hit or cumul_hit:
                kind = "hard" if hard_closed else "soft"
                via = "window" if window_hit else "cumul"
                print(
                    f"  policy failed_close ({kind}/{via}) obs_grip={gripper:.3f} "
                    f"near={near_cube} min_dist={min_dist:.3f} "
                    f"closed_in_window={sum(recent_closed)}/{len(recent_closed)} "
                    f"near_closed_ticks={near_closed_ticks} frames={policy_frames}",
                    flush=True,
                )
                return "failed_close", policy_frames
            if (sleep_t := control_interval - (time.perf_counter() - loop_start)) > 0:
                precise_sleep(sleep_t)
    finally:
        engine.pause()
    print(
        f"  policy timeout obs_grip=[{obs_grip_min:.3f},{obs_grip_max:.3f}] "
        f"cmd_grip=[{cmd_grip_min:.3f},{cmd_grip_max:.3f}] "
        f"seen_open={seen_open} near_last={near_cube} min_dist={min_dist:.3f} "
        f"closed_in_window={sum(recent_closed)}/{len(recent_closed)} "
        f"near_closed_ticks={near_closed_ticks} frames={policy_frames}",
        flush=True,
    )
    return "timeout", policy_frames


def _count_existing(recorder: yaw.ScriptedRecorder, style: str) -> int:
    existing = 0
    if recorder.spawn_log.is_file():
        for line in recorder.spawn_log.read_text().splitlines():
            row = json.loads(line)
            if row.get("success") and row.get("recovery_style") == style:
                existing += 1
    return existing


def main() -> int:
    from lerobot.rollout import build_rollout_context
    from lerobot.rollout.strategies.base import BaseStrategy
    from lerobot.utils.process import ProcessSignalHandler

    args = _parse_args()
    style = _style(args.mode)
    if args.register_v2:
        # Side tree must be on PYTHONPATH (HF LeRobot PR #3967).
        import lerobot.policies.lingbot_vla_v2.configuration_lingbot_vla_v2  # noqa: F401

        print("registered lingbot_vla_v2 policy config", flush=True)
    if not args.policy_path.is_dir():
        print(f"policy not found: {args.policy_path}", file=sys.stderr)
        return 2
    if not args.dataset_root.is_dir():
        print(f"dataset root missing (copy base dataset first): {args.dataset_root}", file=sys.stderr)
        return 2

    cfg = _rollout_config(args)
    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    robot = ctx.hardware.robot_wrapper
    inner = getattr(robot, "inner", robot)
    _expand_rollout_state_features(ctx, inner)
    _fix_preprocessor_normalizer_stats(
        ctx,
        {
            "stats_dataset_repo_id": args.stats_repo_id,
            "stats_dataset_root": str(args.stats_root),
            "state_dim": 15,
        },
        cfg.rename_map,
    )
    strategy = BaseStrategy(cfg.strategy)
    strategy.setup(ctx)
    recorder = yaw.ScriptedRecorder(
        inner,
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
    )
    existing = _count_existing(recorder, style)
    print(
        f"policy DAgger mode={args.mode} style={style} target={args.target} "
        f"existing={existing} policy={args.policy_path} dataset={args.dataset_root}",
        flush=True,
    )
    saved = existing
    attempts = 0
    try:
        while saved < args.target and attempts < args.max_attempts:
            inner.reset_episode(attempts)
            strategy._engine.reset()
            strategy._interpolator.reset()
            block = inner.get_block_position()
            initial_z = block[2] if block is not None else 0.015
            cx, cy, cz, cube_yaw = yaw._block_pose(inner)

            if args.mode == "full":
                recorder.begin_episode(
                    {
                        "spawn": [cx, cy, cz],
                        "yaw_deg": cube_yaw,
                        "yaw_bin": yaw._yaw_bin(cube_yaw),
                        "xy_cell": yaw._xy_cell(cx, cy),
                        "targeted_recovery": True,
                        "recovery_style": style,
                        "policy_checkpoint": args.policy_tag,
                        "intervention": False,
                    }
                )

            outcome, policy_frames = _policy_outcome(
                strategy,
                ctx,
                initial_z,
                recorder=recorder if args.mode == "full" else None,
                closed_hold_ticks=int(args.closed_hold_ticks),
            )
            print(
                f"attempt {attempts} policy outcome={outcome} "
                f"yaw={cube_yaw:+.1f} policy_frames={policy_frames}",
                flush=True,
            )
            attempts += 1

            if outcome != "failed_close":
                if args.mode == "full":
                    # Drop home→success / timeout buffers; we only keep fail→recover.
                    recorder.end_episode(success=False, episode=attempts)
                continue

            success = yaw.recover_and_place(
                inner,
                recorder,
                attempts,
                {
                    "spawn": [cx, cy, cz],
                    "yaw_deg": cube_yaw,
                    "yaw_bin": yaw._yaw_bin(cube_yaw),
                    "xy_cell": yaw._xy_cell(cx, cy),
                    "targeted_recovery": True,
                    "recovery_style": style,
                    "policy_checkpoint": args.policy_tag,
                    "intervention": True,
                    "intervention_start_frame": policy_frames,
                    "policy_frames": policy_frames,
                },
                continue_episode=(args.mode == "full"),
            )
            if success:
                saved += 1
            print(f"  recovery saved={saved}/{args.target} attempts={attempts}", flush=True)
    finally:
        recorder.close()
        inner.disconnect()
    print(f"policy DAgger done mode={args.mode} saved={saved} attempts={attempts}")
    return 0 if saved >= args.target else 1


if __name__ == "__main__":
    raise SystemExit(main())
