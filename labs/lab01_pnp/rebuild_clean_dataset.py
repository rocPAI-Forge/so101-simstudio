#!/usr/bin/env python3
"""Copy selected episodes into a new LeRobot v3 dataset, re-encoding videos.

Frames are decoded at parquet timestamps and written again, so the new videos
hold exactly one frame per parquet row. Stray frames that trail some source
episodes in the video files are dropped. The source dataset is not modified.

  .venv-rocm/bin/python labs/lab01_pnp/rebuild_clean_dataset.py \
    --src datasets/so101-simstudio-lab01-pnp-agent-yaw-v2 \
    --dst datasets/so101-simstudio-lab01-pnp-agent-yaw-v2-clean \
    --repo-id alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v2-clean \
    --episodes 0-23,25-66,68-79,163-177
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

AUTO_KEYS = {"index", "episode_index", "frame_index", "timestamp", "task_index", "task"}


def parse_ranges(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--episodes", required=True, help="e.g. 0-23,25-66,163-177")
    args = ap.parse_args()

    if args.dst.exists():
        raise SystemExit(f"{args.dst} exists; refusing to overwrite")
    keep = parse_ranges(args.episodes)

    src = LeRobotDataset("local/src", root=args.src, video_backend="pyav")
    features = {k: v for k, v in src.meta.features.items() if k not in AUTO_KEYS}
    video_keys = set(src.meta.video_keys)
    dst = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=src.meta.fps,
        features=features,
        root=args.dst,
        robot_type=src.meta.robot_type,
        use_videos=True,
        image_writer_threads=4,
    )

    ep_from = src.meta.episodes["dataset_from_index"]
    ep_to = src.meta.episodes["dataset_to_index"]
    mapping = []
    for new_idx, old in enumerate(keep):
        for i in range(int(ep_from[old]), int(ep_to[old])):
            item = src[i]
            frame = {"task": item["task"]}
            for key in features:
                val = item[key]
                if isinstance(val, torch.Tensor):
                    val = val.numpy()
                if key in video_keys:
                    val = (np.clip(val.transpose(1, 2, 0), 0.0, 1.0) * 255).round().astype(np.uint8)
                frame[key] = val
            dst.add_frame(frame)
        dst.save_episode()
        mapping.append({"new_episode_index": new_idx, "source_episode_index": old})
        print(f"episode {old} -> {new_idx} ({int(ep_to[old]) - int(ep_from[old])} frames)", flush=True)
    dst.finalize()

    (args.dst / "source_episodes.json").write_text(json.dumps(mapping, indent=1))
    print(f"done: {len(keep)} episodes -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
