"""Multi-episode replay for SO-101 MuJoCo.

Replays all (or a subset of) episodes from a recorded dataset sequentially.
Wraps lerobot_replay's single-episode replay in a loop.
"""

import logging
import sys
from pathlib import Path

import yaml
from lerobot.datasets import LeRobotDataset
from lerobot.utils.utils import init_logging

from so101_mujoco_teleop.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401


def main():
    # Parse --config from argv
    config_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--config" and i + 1 < len(sys.argv):
            config_path = Path(sys.argv[i + 1])
            break

    if config_path is None:
        print("Usage: python -m so101_mujoco_teleop.scripts.replay_multi --config <yaml>")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    init_logging()

    ds_cfg = cfg["dataset"]
    repo_id = ds_cfg["repo_id"]
    root = ds_cfg.get("root")
    episodes_cfg = ds_cfg.get("episodes", "all")

    # Determine episode list
    if episodes_cfg == "all":
        full_ds = LeRobotDataset(repo_id, root=root)
        episodes = list(range(full_ds.num_episodes))
        del full_ds
    else:
        episodes = episodes_cfg

    logging.info(f"Episodes to replay: {episodes}")

    # Build base argv for lerobot_replay (without --config, we inject per-episode)
    base_argv = []
    for i, arg in enumerate(sys.argv):
        if arg == "--config":
            continue
        if i > 0 and sys.argv[i - 1] == "--config":
            continue
        base_argv.append(arg)

    # Replay each episode by invoking lerobot_replay with --dataset.episode=N
    for ep in episodes:
        logging.info(f"--- Replaying episode {ep} ---")
        sys.argv = base_argv + [f"--dataset.episode={ep}"]

        from lerobot.scripts.lerobot_replay import replay

        replay()

    logging.info("All episodes replayed.")


if __name__ == "__main__":
    main()
