"""Visualize recorded dataset using LeRobot's built-in Rerun viewer.

Usage:
    .venv-rocm/bin/python -m simstudio.scripts.dataset_viz \
        --repo-id alexhegit/so101_mujoco_leader_test \
        --root ./datasets/leader-test \
        --episode 0

    # Save .rrd and open manually (no spawn):
    .venv-rocm/bin/python -m simstudio.scripts.dataset_viz \
        --repo-id alexhegit/so101-simstudio-pnp \
        --root ./datasets/so101-simstudio-pnp \
        --episode 0 --save --output-dir ./outputs/viz
    .venv-rocm/bin/rerun ./outputs/viz/alexhegit_so101-simstudio-pnp_episode_0.rrd
"""

import argparse
import os
import sys
from pathlib import Path

from lerobot.datasets import LeRobotDataset
from lerobot.scripts.lerobot_dataset_viz import visualize_dataset
from lerobot.utils.import_utils import register_third_party_plugins


def _ensure_rerun_viewer_on_path() -> None:
    """Rerun's spawn() looks for the `rerun` CLI on PATH, not next to sys.executable."""
    venv_bin = Path(sys.executable).resolve().parent
    rerun_cli = venv_bin / "rerun"
    if rerun_cli.is_file():
        os.environ["PATH"] = f"{venv_bin}{os.pathsep}{os.environ.get('PATH', '')}"


def main():
    register_third_party_plugins()

    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset")
    parser.add_argument("--repo-id", required=True, help="Dataset repo ID")
    parser.add_argument("--root", default=None, help="Dataset root directory")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--display-mode", default="rerun", help="rerun or foxglove")
    parser.add_argument(
        "--mode",
        default="local",
        choices=["local", "distant"],
        help="Rerun local spawn vs distant web server",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Write .rrd to --output-dir instead of spawning the viewer",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for .rrd when --save is set",
    )
    args = parser.parse_args()

    if args.save and args.output_dir is None:
        parser.error("--save requires --output-dir")

    _ensure_rerun_viewer_on_path()

    print(f"Loading dataset: {args.repo_id}")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        episodes=[args.episode],
        root=args.root,
        video_backend="pyav",
    )

    print(f"Episode {args.episode}: {dataset.num_frames} frames")
    if args.save:
        print(f"Saving Rerun recording to {args.output_dir} ...")
    else:
        print(f"Launching {args.display_mode} viewer...")

    rrd_path = visualize_dataset(
        dataset,
        episode_index=args.episode,
        display_mode=args.display_mode,
        mode=args.mode,
        save=args.save,
        output_dir=args.output_dir,
    )
    if rrd_path is not None:
        print(f"Saved: {rrd_path}")
        print(f"Open with: {Path(sys.executable).resolve().parent / 'rerun'} {rrd_path}")


if __name__ == "__main__":
    main()
