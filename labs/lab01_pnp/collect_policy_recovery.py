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
    p.add_argument("--n-action-steps", type=int, default=50)
    p.add_argument(
        "--policy-tag",
        default="lab01_pnp_act_agent_yaw_v3/080000",
        help="Stored in spawn metadata only.",
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
        f"--policy.n_action_steps={args.n_action_steps}",
        "--robot.reset_arm=home",
        "--robot.ee_site_name=gripperframe",
        "--robot.home_gripper=0.85",
        "--robot.render_window=false",
        "--display_data=false",
        "--play_sounds=false",
        "--fps=20",
    ]
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
    control_interval = interpolator.get_control_interval(ctx.runtime.cfg.fps)
    seen_open = False
    closed_low = 0
    policy_frames = 0
    engine.resume()
    try:
        for _tick in range(500):
            loop_start = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = strategy._process_observation_and_notify(ctx.processors, obs)
            sent = send_next_action(obs_processed, obs, ctx, interpolator)
            if recorder is not None and sent is not None:
                recorder.on_tick(obs, sent)
                policy_frames += 1
            gripper = float((sent or {}).get("gripper.pos", 1.0))
            block = inner.get_block_position()
            lifted = block is not None and block[2] >= initial_z + 0.02
            if gripper >= 0.45:
                seen_open = True
                closed_low = 0
            closed = seen_open and gripper <= 0.15
            if closed and lifted:
                return "lifted", policy_frames
            if closed and not lifted:
                closed_low += 1
                if closed_low >= 40:
                    return "failed_close", policy_frames
            else:
                closed_low = 0
            if (sleep_t := control_interval - (time.perf_counter() - loop_start)) > 0:
                precise_sleep(sleep_t)
    finally:
        engine.pause()
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
    if not args.policy_path.is_dir():
        print(f"policy not found: {args.policy_path}", file=sys.stderr)
        return 2
    if not args.dataset_root.is_dir():
        print(f"dataset root missing (copy v3 first): {args.dataset_root}", file=sys.stderr)
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
