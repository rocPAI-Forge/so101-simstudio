"""Sim2sim closed-loop policy evaluation in MuJoCo.

Runs a trained policy for multiple episodes with random cube resets,
reports per-episode success (cube in container) and aggregate success rate.

Uses LeRobot's rollout context (policy + pre/post processors + inference
engine) but owns the multi-episode loop so no ``rollout_`` dataset is required.
"""

import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml


def _preselect_mujoco_gl_eval(argv: list[str]) -> None:
    """Use EGL offscreen rendering when GLFW window is disabled or DISPLAY is missing."""
    headless = not os.environ.get("DISPLAY")
    for i, arg in enumerate(argv):
        if arg == "--robot.render_window" and i + 1 < len(argv):
            headless = headless or argv[i + 1].lower() in ("false", "0")
        elif arg.startswith("--robot.render_window="):
            headless = headless or arg.split("=", 1)[1].lower() in ("false", "0")
    if headless:
        os.environ["MUJOCO_GL"] = "egl"


_preselect_mujoco_gl_eval(sys.argv[1:])

from lerobot.configs import parser  # noqa: E402
from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context  # noqa: E402
from lerobot.rollout.strategies.base import BaseStrategy  # noqa: E402
from lerobot.rollout.strategies.core import send_next_action  # noqa: E402
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
from lerobot.utils.process import ProcessSignalHandler  # noqa: E402
from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
from lerobot.utils.utils import init_logging, log_say  # noqa: E402
from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization  # noqa: E402

from simstudio.common.eval_success import check_pick_success  # noqa: E402
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401,E402
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EVAL = {
    "num_episodes": 10,
    "episode_time_s": 60.0,
    "reset_time_s": 3.0,
    "stats_dataset_repo_id": "alexhegit/so101-simstudio-pnp",
    "stats_dataset_root": "./datasets/so101-simstudio-pnp",
    "state_dim": 6,
}

_EVAL_SETTINGS: dict[str, Any] = {}


def _config_path_from_argv(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    print("Usage: python -m simstudio.scripts.eval --config <yaml> --policy.path <checkpoint>")
    sys.exit(1)


def _parse_eval_overrides(argv: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--eval."):
            i += 1
            continue
        field = arg.removeprefix("--eval.")
        if "=" in field:
            name, value = field.split("=", 1)
        elif i + 1 < len(argv):
            name = field
            value = argv[i + 1]
            i += 1
        else:
            i += 1
            continue
        if name == "num_episodes":
            overrides[name] = int(value)
        elif name in ("episode_time_s", "reset_time_s"):
            overrides[name] = float(value)
        i += 1
    return overrides


def _prepare_argv_for_rollout_parser() -> None:
    """Map ``--config`` to LeRobot ``--config_path`` and drop the ``eval:`` yaml block."""
    argv = sys.argv[1:]
    config_path = _config_path_from_argv(argv)
    eval_overrides = _parse_eval_overrides(argv)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    global _EVAL_SETTINGS
    _EVAL_SETTINGS = {**DEFAULT_EVAL, **raw.pop("eval", {}), **eval_overrides}
    rollout_doc = raw
    rollout_doc.setdefault("strategy", {"type": "base"})

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(rollout_doc, tmp, sort_keys=False)
    tmp.close()

    new_argv = [sys.argv[0], f"--config_path={tmp.name}"]
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--config",):
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        if arg.startswith("--eval."):
            if "=" not in arg:
                skip_next = True
            continue
        new_argv.append(arg)
    sys.argv = new_argv


def _run_episode(
    strategy: BaseStrategy,
    ctx: Any,
    episode_time_s: float,
) -> None:
    cfg = ctx.runtime.cfg
    robot = ctx.hardware.robot_wrapper
    engine = strategy._engine
    interpolator = strategy._interpolator
    control_interval = interpolator.get_control_interval(cfg.fps)

    start = time.perf_counter()
    engine.resume()
    while not ctx.runtime.shutdown_event.is_set():
        if time.perf_counter() - start >= episode_time_s:
            break

        loop_start = time.perf_counter()
        obs = robot.get_observation()
        obs_processed = strategy._process_observation_and_notify(ctx.processors, obs)

        if strategy._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
            continue

        send_next_action(obs_processed, obs, ctx, interpolator)

        dt = time.perf_counter() - loop_start
        if (sleep_t := control_interval - dt) > 0:
            precise_sleep(sleep_t)


def _reset_pause(robot: SO101MujocoRobot, reset_time_s: float, fps: float) -> None:
    steps = max(int(reset_time_s * fps), 1)
    for _ in range(steps):
        if robot.config.render_window:
            robot._render_glfw()
        precise_sleep(1.0 / fps)


def _fix_preprocessor_normalizer_stats(ctx: Any, eval_settings: dict[str, Any], rename_map: dict) -> None:
    """Align normalizer ``observation.state`` stats with the 6-dim policy input.

    Training checkpoints may embed 15-dim dataset stats (pos+vel+ee) while rollout
    feeds 6 joint positions — slice stats to match.
    """
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.rename_processor import rename_stats

    repo_id = eval_settings["stats_dataset_repo_id"]
    root = eval_settings.get("stats_dataset_root")
    state_dim = int(eval_settings.get("state_dim", 6))

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    stats = rename_stats(ds_meta.stats, rename_map)
    if "observation.state" in stats:
        stats["observation.state"] = {
            k: v[:state_dim] for k, v in stats["observation.state"].items()
        }

    for step in ctx.policy.preprocessor.steps:
        if isinstance(step, NormalizerProcessorStep):
            step.stats = stats
            step.to(device=step.device)
            logger.info(
                "Patched normalizer stats from %s (observation.state dim=%d)",
                repo_id,
                state_dim,
            )
            return
    logger.warning("NormalizerProcessorStep not found; skipping stats patch")


@parser.wrap()
def eval_main(cfg: RolloutConfig) -> None:
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError("eval.py requires strategy.type=base (multi-episode loop is built-in)")

    eval_settings = _EVAL_SETTINGS
    num_episodes = int(eval_settings["num_episodes"])
    episode_time_s = float(eval_settings["episode_time_s"])
    reset_time_s = float(eval_settings["reset_time_s"])

    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="eval",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    _fix_preprocessor_normalizer_stats(ctx, eval_settings, cfg.rename_map)
    robot = ctx.hardware.robot_wrapper
    inner = getattr(robot, "inner", robot)
    if not isinstance(inner, SO101MujocoRobot):
        raise TypeError(f"eval.py expects so101_mujoco robot, got {type(inner)}")

    strategy = BaseStrategy(cfg.strategy)
    results: list[dict[str, Any]] = []

    logger.info(
        "Starting sim2sim eval: %d episodes × %.0fs @ %.0f Hz | policy=%s",
        num_episodes,
        episode_time_s,
        cfg.fps,
        cfg.policy.pretrained_path,
    )
    log_say(f"Sim2sim eval: {num_episodes} episodes", cfg.play_sounds)

    try:
        strategy.setup(ctx)
        for ep in range(num_episodes):
            if ctx.runtime.shutdown_event.is_set():
                break

            log_say(f"Eval episode {ep + 1} of {num_episodes}", cfg.play_sounds)
            if getattr(inner.config, "reset_mode", "manual") == "auto":
                inner.reset_episode(ep)

            strategy._engine.reset()
            strategy._interpolator.reset()
            _run_episode(strategy, ctx, episode_time_s)

            block_pos = inner.get_block_position()
            success = check_pick_success(block_pos)
            results.append({"episode": ep, "success": success, "block_pos": block_pos})
            status = "SUCCESS" if success else "fail"
            pos_str = f"{block_pos}" if block_pos else "unknown"
            logger.info("Episode %d: %s (block=%s)", ep, status, pos_str)
            print(f"Episode {ep + 1}/{num_episodes}: {status}  block={pos_str}")

            if ep < num_episodes - 1 and reset_time_s > 0:
                _reset_pause(inner, reset_time_s, cfg.fps)
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization()

    if results:
        n_success = sum(1 for r in results if r["success"])
        rate = 100.0 * n_success / len(results)
        summary = f"Eval complete: {n_success}/{len(results)} success ({rate:.0f}%)"
        logger.info(summary)
        print(summary)


def main() -> None:
    register_third_party_plugins()
    init_logging()
    eval_main()


if __name__ == "__main__":
    _prepare_argv_for_rollout_parser()
    main()
