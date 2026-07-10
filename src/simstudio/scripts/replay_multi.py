"""Multi-episode replay for SO-101 MuJoCo.

Replays all (or a subset of) episodes from a recorded dataset sequentially
in a single MuJoCo session (one GLFW window for the full run).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import draccus
import yaml
from lerobot.configs.parser import parse_arg
from lerobot.datasets import LeRobotDataset
from lerobot.processor import make_default_robot_action_processor
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_replay import ReplayConfig
from lerobot.utils.constants import ACTION
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say

from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot


def _config_path_from_argv() -> Path:
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1])
    print("Usage: python -m simstudio.scripts.replay_multi --config <yaml>")
    sys.exit(1)


def _cli_args_for_draccus() -> list[str]:
    """Drop --config and episode-selection flags; keep dataset/robot overrides."""
    filtered: list[str] = []
    skip_next = False
    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--config":
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        if arg in ("--dataset.episode", "--dataset.episodes"):
            skip_next = True
            continue
        if arg.startswith("--dataset.episode=") or arg.startswith("--dataset.episodes="):
            continue
        filtered.append(arg)
    return filtered


def _parse_replay_config(config_path: Path) -> ReplayConfig:
    register_third_party_plugins()
    return draccus.parse(
        config_class=ReplayConfig,
        config_path=config_path,
        args=_cli_args_for_draccus() + ["--dataset.episode=0"],
    )


def _episode_list(repo_id: str, root: str | Path | None, episodes_cfg: str | list[int]) -> list[int]:
    if episodes_cfg == "all":
        return list(range(LeRobotDataset(repo_id, root=root).meta.total_episodes))
    return [int(ep) for ep in episodes_cfg]


def main() -> None:
    config_path = _config_path_from_argv()
    cfg = _parse_replay_config(config_path)

    with open(config_path) as f:
        yaml_cfg = yaml.safe_load(f)

    episodes_cfg = parse_arg("--dataset.episodes") or yaml_cfg.get("dataset", {}).get("episodes", "all")
    if parse_arg("--dataset.root"):
        cfg.dataset.root = parse_arg("--dataset.root")
    if parse_arg("--dataset.repo_id"):
        cfg.dataset.repo_id = parse_arg("--dataset.repo_id")

    episodes = _episode_list(cfg.dataset.repo_id, cfg.dataset.root, episodes_cfg)
    init_logging()
    logging.info("Replaying episodes %s from %s", episodes, cfg.dataset.root)

    robot_action_processor = make_default_robot_action_processor()
    robot = make_robot_from_config(cfg.robot)
    robot.connect()

    try:
        for ep in episodes:
            log_say(f"Replaying episode {ep}", cfg.play_sounds, blocking=True)
            if isinstance(robot, SO101MujocoRobot) and getattr(robot.config, "reset_mode", "manual") == "auto":
                robot.reset_episode(ep)

            dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root, episodes=[ep])
            actions = dataset.select_columns(ACTION)

            for idx in range(dataset.num_frames):
                start_episode_t = time.perf_counter()

                action_array = actions[idx][ACTION]
                action = {
                    name: action_array[i] for i, name in enumerate(dataset.features[ACTION]["names"])
                }

                robot_obs = robot.get_observation()
                processed_action = robot_action_processor((action, robot_obs))
                _ = robot.send_action(processed_action)

                dt_s = time.perf_counter() - start_episode_t
                precise_sleep(max(1 / dataset.fps - dt_s, 0.0))
    finally:
        robot.disconnect()

    logging.info("All episodes replayed.")


if __name__ == "__main__":
    main()
