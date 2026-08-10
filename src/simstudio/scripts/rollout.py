"""Project wrapper around LeRobot's ``lerobot-rollout`` CLI.

Registers the SO-101 MuJoCo robot plugin and hooks sim episode reset
(``reset_episode``) for base / episodic rollout strategies.
"""

from __future__ import annotations

import functools
import logging
import os
import sys
from typing import Any

def _preselect_mujoco_gl_rollout(argv: list[str]) -> None:
    """Use EGL when GLFW window is disabled or DISPLAY is missing."""
    headless = not os.environ.get("DISPLAY")
    for i, arg in enumerate(argv):
        if arg == "--robot.render_window" and i + 1 < len(argv):
            headless = headless or argv[i + 1].lower() in ("false", "0")
        elif arg.startswith("--robot.render_window="):
            headless = headless or arg.split("=", 1)[1].lower() in ("false", "0")
    if headless:
        os.environ["MUJOCO_GL"] = "egl"
    else:
        from simstudio.scripts.record import _preselect_mujoco_gl

        _preselect_mujoco_gl(argv)


_preselect_mujoco_gl_rollout(sys.argv[1:])

from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401,E402

logger = logging.getLogger(__name__)


def _robot_inner(robot: Any) -> Any:
    return getattr(robot, "inner", robot)


def _maybe_reset_sim(robot: Any, episode_index: int) -> None:
    inner = _robot_inner(robot)
    if not hasattr(inner, "reset_episode"):
        return
    reset_mode = getattr(getattr(inner, "config", None), "reset_mode", "manual")
    if reset_mode != "auto":
        return
    logger.info("Auto-resetting sim for rollout episode %s", episode_index)
    inner.reset_episode(episode_index)


def _patch_rollout_sim_reset() -> None:
    """Hook LeRobot rollout strategies to reset MuJoCo arm + cube between episodes."""
    from lerobot.rollout.strategies import base as base_mod
    from lerobot.rollout.strategies import episodic as episodic_mod

    if getattr(episodic_mod.EpisodicStrategy._policy_loop, "_so101_reset_patched", False):
        return

    orig_policy_loop = episodic_mod.EpisodicStrategy._policy_loop

    @functools.wraps(orig_policy_loop)
    def _patched_policy_loop(
        self,
        ctx: Any,
        robot: Any,
        events: Any,
        features: Any,
        fps: float,
        control_time_s: float,
        dataset: Any,
        single_task: str,
    ):
        episode_index = dataset.num_episodes if dataset is not None else 0
        _maybe_reset_sim(robot, episode_index)
        return orig_policy_loop(
            self, ctx, robot, events, features, fps, control_time_s, dataset, single_task
        )

    _patched_policy_loop._so101_reset_patched = True
    episodic_mod.EpisodicStrategy._policy_loop = _patched_policy_loop

    orig_base_setup = base_mod.BaseStrategy.setup

    @functools.wraps(orig_base_setup)
    def _patched_base_setup(self, ctx: Any) -> None:
        orig_base_setup(self, ctx)
        _maybe_reset_sim(ctx.hardware.robot_wrapper, 0)

    _patched_base_setup._so101_reset_patched = True
    base_mod.BaseStrategy.setup = _patched_base_setup


_patch_rollout_sim_reset()


def main() -> None:
    argv = ["lerobot_rollout.py", *sys.argv[1:]]
    sys.argv = argv

    from lerobot.scripts.lerobot_rollout import main as lerobot_rollout_main

    lerobot_rollout_main()


if __name__ == "__main__":
    main()
