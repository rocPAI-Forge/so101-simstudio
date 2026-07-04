"""Visualize recorded dataset episodes.

Usage:
    .venv-rocm/bin/python -m so101_mujoco_teleop.scripts.dataset_viz \
        --repo-id alexhegit/so101_mujoco_leader_test \
        --root ./datasets/leader-test \
        --episode 0
"""

import argparse

from lerobot.datasets import LeRobotDataset
from lerobot.utils.import_utils import register_third_party_plugins


def main():
    register_third_party_plugins()

    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID")
    parser.add_argument("--root", default=None, help="Dataset root directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    args = parser.parse_args()

    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=[args.episode],
        root=args.root,
        video_backend="pyav",
    )

    print("\nDataset info:")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Features: {list(dataset.features.keys())}")

    if "action" in dataset.features:
        names = dataset.features["action"].get("names", [])
        print(f"  Action names: {names}")

    # Sample a few frames to show action stats
    import torch

    n_samples = min(100, dataset.num_frames)
    actions = []
    for i in range(n_samples):
        frame = dataset[i]
        if "action" in frame:
            actions.append(frame["action"])

    if actions:
        actions_tensor = torch.stack(actions)
        print(f"\nAction stats (first {n_samples} frames):")
        names = dataset.features.get("action", {}).get(
            "names", [f"dim_{i}" for i in range(actions_tensor.shape[1])]
        )
        for i, name in enumerate(names):
            vals = actions_tensor[:, i]
            print(f"  {name}: min={vals.min():.4f}, max={vals.max():.4f}, mean={vals.mean():.4f}")

    # Try visualization with rerun
    try:
        from lerobot.scripts.lerobot_dataset_viz import visualize_dataset

        print("\nLaunching Rerun viewer...")
        visualize_dataset(dataset, episode_index=args.episode, display_mode="rerun")
    except Exception as e:
        print(f"\nRerun visualization unavailable: {e}")
        print("Showing basic stats only.")


if __name__ == "__main__":
    main()
