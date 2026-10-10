#!/usr/bin/env python3
"""Teacher-forced SmolVLA probe around expert gripper-release transitions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from probe_visual_grounding import (
    _normalization_name,
    _patch_processor_state_stats,
    _rename_map,
    _unnormalize,
)


def _release_contexts(
    root: Path,
    first_episode: int,
    episodes: int,
    lead_frames: int,
) -> list[dict[str, int]]:
    tables = [
        pq.read_table(path, columns=["episode_index", "frame_index", "index", "action"])
        for path in sorted((root / "data").glob("**/*.parquet"))
    ]
    table = pa.concat_tables(tables)
    episode_index = np.asarray(table["episode_index"])
    frame_index = np.asarray(table["frame_index"])
    index = np.asarray(table["index"])
    action = np.stack(table["action"].to_pylist())

    contexts = []
    for episode in range(first_episode, first_episode + episodes):
        mask = episode_index == episode
        frames = frame_index[mask]
        indices = index[mask]
        gripper = action[mask, -1]
        rising = np.flatnonzero((np.diff(gripper) > 0.01) & (gripper[1:] > 0.0)) + 1
        if not len(rising):
            continue
        release_offset = int(rising[0])
        context_offset = max(0, release_offset - lead_frames)
        contexts.append(
            {
                "episode": episode,
                "release_frame": int(frames[release_offset]),
                "context_frame": int(frames[context_offset]),
                "index": int(indices[context_offset]),
                "release_offset": release_offset - context_offset,
            }
        )
    return contexts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--stats-root", type=Path)
    parser.add_argument(
        "--repo-id",
        default="alexhegit/so101-simstudio-lab01-pnp-agent-yaw-v4",
    )
    parser.add_argument("--first-episode", type=int, default=140)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--lead-frames", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    stats_root = args.stats_root or args.dataset_root

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    if cfg.type != "smolvla":
        raise ValueError(f"This focused probe expects SmolVLA, got {cfg.type}")
    cfg.pretrained_path = str(args.checkpoint)
    cfg.device = "cuda"
    horizon = int(getattr(cfg, "action_horizon", 0) or getattr(cfg, "chunk_size", 50))
    state_dim = int(cfg.input_features["observation.state"].shape[0])
    rename = _rename_map(cfg.type)
    mode = _normalization_name(cfg)

    contexts = _release_contexts(
        args.dataset_root, args.first_episode, args.episodes, args.lead_frames
    )
    if not contexts:
        raise RuntimeError("No expert release transitions found")

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        video_backend="pyav",
        delta_timestamps={"action": [i / 20 for i in range(horizon)]},
    )
    stats = json.loads((stats_root / "meta/stats.json").read_text())
    renamed_stats = rename_stats(stats, rename)
    renamed_stats["observation.state"] = {
        key: value[:state_dim] for key, value in renamed_stats["observation.state"].items()
    }

    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint, config=cfg).to("cuda")
    policy.eval()
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.checkpoint,
        dataset_stats=renamed_stats,
        preprocessor_overrides={
            "device_processor": {"device": "cuda"},
            "rename_observations_processor": {"rename_map": rename},
        },
    )
    _patch_processor_state_stats(preprocessor, renamed_stats["observation.state"])
    src_by_dst = {dst: src for src, dst in rename.items()}

    rows: list[dict[str, Any]] = []
    for context in contexts:
        item = dataset[context["index"]]
        batch: dict[str, Any] = {
            "observation.state": item["observation.state"][:state_dim].unsqueeze(0),
            "task": item.get("task", "Pick up the cube and place it in the box."),
        }
        for dst in cfg.image_features:
            src = src_by_dst.get(dst, dst)
            if src not in item:
                continue
            batch[src] = item[src].unsqueeze(0)

        policy.reset()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            prediction = policy.predict_action_chunk(preprocessor(batch))[0]
        prediction = _unnormalize(
            prediction.float().cpu().numpy().astype(np.float64),
            stats["action"],
            mode,
        )[:, :6]
        expert = item["action"].numpy().astype(np.float64)[: len(prediction), :6]
        prediction = prediction[: len(expert)]
        pred_gripper = prediction[:, -1]
        expert_gripper = expert[:, -1]
        release_offset = context["release_offset"]
        post = slice(release_offset, None)
        rows.append(
            {
                **context,
                "expert_gripper_start": float(expert_gripper[0]),
                "expert_gripper_max": float(expert_gripper[post].max()),
                "pred_gripper_start": float(pred_gripper[0]),
                "pred_gripper_max_after_release": float(pred_gripper[post].max()),
                "pred_gripper_end": float(pred_gripper[-1]),
                "pred_opens_above_0_3": bool(np.any(pred_gripper[post] > 0.3)),
                "pred_opens_above_0_6": bool(np.any(pred_gripper[post] > 0.6)),
                "gripper_rmse": float(np.sqrt(np.mean((pred_gripper - expert_gripper) ** 2))),
                "pred_gripper": pred_gripper.tolist(),
                "expert_gripper": expert_gripper.tolist(),
            }
        )

    result = {
        "checkpoint": str(args.checkpoint),
        "dataset_root": str(args.dataset_root),
        "stats_root": str(stats_root),
        "scope": "in-sample teacher-forced diagnostic on expert release contexts",
        "horizon": horizon,
        "lead_frames": args.lead_frames,
        "samples": len(rows),
        "summary": {
            "opens_above_0_3": int(sum(row["pred_opens_above_0_3"] for row in rows)),
            "opens_above_0_6": int(sum(row["pred_opens_above_0_6"] for row in rows)),
            "mean_pred_max_after_release": float(
                np.mean([row["pred_gripper_max_after_release"] for row in rows])
            ),
            "mean_pred_end": float(np.mean([row["pred_gripper_end"] for row in rows])),
            "mean_gripper_rmse": float(np.mean([row["gripper_rmse"] for row in rows])),
        },
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "rows": "<omitted>"}, indent=2))


if __name__ == "__main__":
    main()
