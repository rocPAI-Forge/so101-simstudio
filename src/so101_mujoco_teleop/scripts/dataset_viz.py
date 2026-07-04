"""Visualize recorded dataset using LeRobot's built-in Rerun viewer.

Usage:
    .venv-rocm/bin/python -m so101_mujoco_teleop.scripts.dataset_viz \
        --repo-id alexhegit/so101_mujoco_leader_test \
        --root ./datasets/leader-test \
        --episode 0
"""

import argparse

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.import_utils import register_third_party_plugins


def main():
    register_third_party_plugins()

    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID")
    parser.add_argument("--root", default=None, help="Dataset root directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--display-mode", default="rerun", help="rerun or foxglove")
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=[args.episode],
        root=args.root,
        video_backend="pyav",
    )

    print(f"Episode {args.episode}: {dataset.num_frames} frames")
    print(f"Launching {args.display_mode} viewer...")

    visualize_dataset(
        dataset,
        episode_index=args.episode,
        display_mode=args.display_mode,
    )


if __name__ == "__main__":
    main()
