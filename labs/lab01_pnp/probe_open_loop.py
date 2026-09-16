#!/usr/bin/env python3
"""Unified teacher-forced action-fidelity probe for Lab 01 policies.

All current checkpoints were trained on all 50 episodes, so these measurements
are explicitly *in-sample diagnostics*, not held-out generalization. They answer
whether a checkpoint can reproduce its own demonstrations at episode start and
at normalized temporal quartiles. A cross-episode image shuffle measures whether
the prediction uses scene pixels beyond proprio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.rename_processor import rename_stats
from probe_visual_grounding import (
    DATASET,
    _normalization_name,
    _patch_processor_state_stats,
    _preserve_vla_jepa_dims,
    _rename_map,
    _unnormalize,
)

PHASES = {"start": 0.0, "quarter": 0.25, "middle": 0.5, "three_quarters": 0.75}


def _episode_bounds(ds: LeRobotDataset, n: int) -> list[tuple[int, int]]:
    episodes = ds.meta.episodes
    starts = None
    ends = None
    for key in ("dataset_from_index", "from_index", "data_index_from"):
        if key in episodes.column_names:
            starts = [int(v) for v in episodes[key][:n]]
            break
    for key in ("dataset_to_index", "to_index", "data_index_to"):
        if key in episodes.column_names:
            ends = [int(v) for v in episodes[key][:n]]
            break
    if starts is None:
        raise RuntimeError(f"No episode start column in {episodes.column_names}")
    if ends is None:
        ends = starts[1:] + [ds.num_frames]
    return list(zip(starts, ends, strict=True))


def _sample_index(start: int, end: int, horizon: int, fraction: float) -> int:
    last = max(start, end - horizon - 1)
    return min(last, start + round(fraction * (last - start)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    raw_cfg = json.loads((args.checkpoint / "config.json").read_text())
    policy_type = raw_cfg["type"]
    if policy_type == "vla_jepa":
        _preserve_vla_jepa_dims()
    if policy_type == "fastwam":
        from simstudio.policies.fastwam_pad import patch_fastwam_pad_action_channels

        patch_fastwam_pad_action_channels(6)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = str(args.checkpoint)
    cfg.device = "cuda"
    if policy_type == "molmoact2":
        cfg.enable_inference_cuda_graph = False
    horizon = int(getattr(cfg, "action_horizon", 0) or getattr(cfg, "chunk_size", 32))
    state_feature = (cfg.input_features or {}).get("observation.state")
    state_dim = int(
        getattr(cfg, "state_dim", 0)
        or (state_feature.shape[0] if state_feature is not None else 0)
        or getattr(cfg, "proprio_dim", 0)
    )
    rename = _rename_map(policy_type)
    mode = _normalization_name(cfg)

    base = LeRobotDataset(
        "alexhegit/so101-simstudio-lab01-pnp",
        root=DATASET,
        video_backend="pyav",
    )
    bounds = _episode_bounds(base, args.episodes)
    ds = LeRobotDataset(
        "alexhegit/so101-simstudio-lab01-pnp",
        root=DATASET,
        video_backend="pyav",
        delta_timestamps={"action": [i / base.meta.fps for i in range(horizon)]},
    )
    stats = base.meta.stats
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
    image_keys = list(cfg.image_features)
    src_by_dst = {dst: src for src, dst in rename.items()}

    def predict(item: dict[str, Any], image_item: dict[str, Any]) -> np.ndarray:
        batch: dict[str, Any] = {
            "observation.state": item["observation.state"][:state_dim].unsqueeze(0),
            "task": item.get("task", "Pick up the cube and place it in the box."),
        }
        for dst in image_keys:
            src = src_by_dst.get(dst, dst)
            if src not in image_item:
                continue
            batch[src] = image_item[src].unsqueeze(0)
        policy.reset()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            pred = policy.predict_action_chunk(preprocessor(batch))[0]
        return _unnormalize(
            pred.float().cpu().numpy().astype(np.float64), stats["action"], mode
        )[:, :6]

    samples: dict[str, list[dict[str, Any]]] = {phase: [] for phase in PHASES}
    aggregate = []
    for phase_name, fraction in PHASES.items():
        for episode, (start, end) in enumerate(bounds):
            idx = _sample_index(start, end, horizon, fraction)
            shuffle_start, shuffle_end = bounds[(episode + 1) % len(bounds)]
            shuffle_idx = _sample_index(shuffle_start, shuffle_end, horizon, fraction)
            item = ds[idx]
            pred = predict(item, item)
            pred_shuffle = predict(item, ds[shuffle_idx])
            gt = item["action"].numpy().astype(np.float64)[: len(pred), :6]
            pred = pred[: len(gt)]
            pred_shuffle = pred_shuffle[: len(gt)]
            current = item["observation.state"][:6].numpy().astype(np.float64)
            hold = np.repeat(current[None], len(gt), axis=0)
            row = {
                "episode": episode,
                "index": idx,
                "rmse_rad": float(np.sqrt(np.mean((pred - gt) ** 2))),
                "hold_rmse_rad": float(np.sqrt(np.mean((hold - gt) ** 2))),
                "shuffled_image_rmse_rad": float(np.sqrt(np.mean((pred_shuffle - gt) ** 2))),
                "prediction_change_from_shuffle_rad": float(
                    np.sqrt(np.mean((pred_shuffle - pred) ** 2))
                ),
                "pred_travel_rad": float(np.max(np.abs(pred - pred[0]))),
                "demo_travel_rad": float(np.max(np.abs(gt - gt[0]))),
                "per_joint_squared_error": np.mean((pred - gt) ** 2, axis=0).tolist(),
                "gripper_pred_mean_rad": float(pred[:, -1].mean()),
                "gripper_demo_mean_rad": float(gt[:, -1].mean()),
            }
            samples[phase_name].append(row)
            aggregate.append(row)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        per_joint = np.asarray([row["per_joint_squared_error"] for row in rows])
        return {
            "samples": len(rows),
            "rmse_rad": float(np.sqrt(np.mean([row["rmse_rad"] ** 2 for row in rows]))),
            "hold_rmse_rad": float(
                np.sqrt(np.mean([row["hold_rmse_rad"] ** 2 for row in rows]))
            ),
            "shuffled_image_rmse_rad": float(
                np.sqrt(np.mean([row["shuffled_image_rmse_rad"] ** 2 for row in rows]))
            ),
            "prediction_change_from_shuffle_rad": float(
                np.sqrt(
                    np.mean([row["prediction_change_from_shuffle_rad"] ** 2 for row in rows])
                )
            ),
            "travel_ratio": float(
                np.mean([row["pred_travel_rad"] for row in rows])
                / (np.mean([row["demo_travel_rad"] for row in rows]) + 1e-12)
            ),
            "per_joint_rmse_rad": np.sqrt(per_joint.mean(axis=0)).tolist(),
            "gripper_pred_mean_rad": float(
                np.mean([row["gripper_pred_mean_rad"] for row in rows])
            ),
            "gripper_demo_mean_rad": float(
                np.mean([row["gripper_demo_mean_rad"] for row in rows])
            ),
        }

    result = {
        "policy": policy_type,
        "checkpoint": str(args.checkpoint),
        "scope": (
            "in-sample teacher-forced diagnostic; all 50 episodes were used for training, "
            "so this is not held-out generalization"
        ),
        "horizon": horizon,
        "state_dim": state_dim,
        "action_normalization": mode,
        "summary": summarize(aggregate),
        "by_phase": {name: summarize(rows) for name, rows in samples.items()},
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({**result, "samples": "<omitted>"}, indent=2))


if __name__ == "__main__":
    main()
