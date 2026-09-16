#!/usr/bin/env python3
"""Build the evidence bundle for Lab 01's cross-policy causal analysis.

This intentionally avoids importing policy implementations: it audits YAML,
checkpoint JSON, parquet action/state data, eval summaries, and probe JSON.
The output is a compact machine-readable input for the runbook and Canvas.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import yaml

REPO = Path(__file__).resolve().parents[2]
LAB = REPO / "labs/lab01_pnp"
DATASET = REPO / "datasets/so101-simstudio-lab01-pnp"


# These are the published reference rows in lab01_pnp.md §6.4. Protocol fields
# are explicit so consumers cannot accidentally present them as one benchmark.
REFERENCE_RUNS = [
    {
        "id": "act15_full",
        "policy": "ACT",
        "variant": "50K 15-D",
        "successes": 32,
        "episodes": 50,
        "spawn": "full",
        "reset_arm": "follow",
        "inference": "sync",
        "n_action_steps": 50,
        "cameras": 3,
        "state_dim": 15,
        "evidence": "runbook",
    },
    {
        "id": "act6_full",
        "policy": "ACT",
        "variant": "50K 6-D",
        "successes": 29,
        "episodes": 50,
        "spawn": "full",
        "reset_arm": "follow",
        "inference": "sync",
        "n_action_steps": 50,
        "cameras": 3,
        "state_dim": 6,
        "evidence": "outputs/eval/act_state6_egl_fullrange.log",
    },
    {
        "id": "smolvla_full",
        "policy": "SmolVLA",
        "variant": "50K",
        "successes": 11,
        "episodes": 50,
        "spawn": "full",
        "reset_arm": "home",
        "inference": "rtc",
        "n_action_steps": 50,
        "cameras": 3,
        "state_dim": 6,
        "evidence": "runbook",
    },
    {
        "id": "molmo15_full",
        "policy": "MolmoAct2",
        "variant": "10K 15-D",
        "successes": 15,
        "episodes": 50,
        "spawn": "full",
        "reset_arm": "home",
        "inference": "sync",
        "n_action_steps": 30,
        "cameras": 2,
        "state_dim": 15,
        "evidence": "outputs/eval/molmoact2_full.log",
    },
    {
        "id": "molmo6_full",
        "policy": "MolmoAct2",
        "variant": "10K 6-D",
        "successes": 27,
        "episodes": 50,
        "spawn": "full",
        "reset_arm": "home",
        "inference": "sync",
        "n_action_steps": 30,
        "cameras": 2,
        "state_dim": 6,
        "evidence": "outputs/eval/molmoact2_state6_full.log",
    },
    {
        "id": "jepa_bc_fixed",
        "policy": "VLA-JEPA",
        "variant": "BC 10K 6-D",
        "successes": 1,
        "episodes": 10,
        "spawn": "fixed",
        "reset_arm": "demo_mean_home",
        "inference": "sync",
        "n_action_steps": 30,
        "cameras": 2,
        "state_dim": 6,
        "evidence": "outputs/eval/vla_jepa_state6_bc_010000_demostart.log",
    },
    {
        "id": "jepa_wm_fixed",
        "policy": "VLA-JEPA",
        "variant": "WM 20K 6-D",
        "successes": 0,
        "episodes": 10,
        "spawn": "fixed",
        "reset_arm": "keyboard_home",
        "inference": "sync",
        "n_action_steps": 7,
        "cameras": 2,
        "state_dim": 6,
        "evidence": "outputs/eval/vla_jepa_state6_fixed.log",
    },
    {
        "id": "fastwam_unfreeze_full",
        "policy": "FastWAM",
        "variant": "pad7 unfreeze-4 8K",
        "successes": 0,
        "episodes": 10,
        "spawn": "full",
        "reset_arm": "keyboard_home",
        "inference": "sync",
        "n_action_steps": 32,
        "cameras": 2,
        "state_dim": 6,
        "evidence": "DORobot OSMesa 10-episode log; local GLFW confirms 0/5",
    },
]


CHECKPOINTS = {
    "act6": "outputs/train/lab01_pnp_act_state6/checkpoints/050000/pretrained_model",
    "act15": "outputs/train/lab01_pnp_act/checkpoints/last/pretrained_model",
    "smolvla": "outputs/train/lab01_pnp_smolvla/checkpoints/050000/pretrained_model",
    "molmo6": "outputs/train/lab01_pnp_molmoact2_state6/checkpoints/010000/pretrained_model",
    "molmo15": "outputs/train/lab01_pnp_molmoact2/checkpoints/010000/pretrained_model",
    "jepa_bc10k": "outputs/train/lab01_pnp_vla_jepa_state6_bc/checkpoints/010000/pretrained_model",
    "jepa_wm20k": "outputs/train/lab01_pnp_vla_jepa_state6/checkpoints/020000/pretrained_model",
    "fastwam": (
        "outputs/train/lab01_pnp_fastwam_pad7_unfreeze4/checkpoints/008000/pretrained_model"
    ),
}


PROBES = {
    "act6": ".tmp/act_state6_visual_probe.json",
    "jepa_bc10k": ".tmp/jepa_bc10k_visual_probe.json",
    "fastwam": ".tmp/fastwam_visual_probe.json",
}


TRAINING_PROTOCOLS = {
    "act6": {
        "steps": 50_000,
        "batch_size": 128,
        "sample_updates": 6_400_000,
        "cameras": ["top", "front", "wrist"],
        "state_dim": 6,
        "action_dim": 6,
        "action_norm": "MEAN_STD",
        "chunk_size": 100,
        "frozen": "none",
        "pretrained": "ImageNet ResNet-18; ACT policy trained on Lab 01",
    },
    "smolvla": {
        "steps": 50_000,
        "batch_size": 64,
        "sample_updates": 3_200_000,
        "cameras": ["top", "front", "wrist"],
        "state_dim": 6,
        "action_dim": 6,
        "action_norm": "MEAN_STD",
        "chunk_size": 50,
        "frozen": "SmolVLM vision-language backbone mostly reused",
        "pretrained": "lerobot/smolvla_base",
    },
    "molmo6": {
        "steps": 10_000,
        "batch_size": 32,
        "sample_updates": 320_000,
        "cameras": ["top", "wrist"],
        "state_dim": 6,
        "action_dim": 6,
        "action_norm": "QUANTILES",
        "chunk_size": 30,
        "frozen": "base VLM with LoRA; action expert trainable",
        "pretrained": "lerobot/MolmoAct2-SO100_101-LeRobot (same 6-joint embodiment)",
    },
    "jepa_bc10k": {
        "steps": 10_000,
        "batch_size": 16,
        "sample_updates": 160_000,
        "cameras": ["top", "wrist"],
        "state_dim": 6,
        "action_dim": 6,
        "action_norm": "MIN_MAX",
        "chunk_size": 30,
        "frozen": "Qwen frozen; action/state projections reinitialized",
        "pretrained": "LIBERO VLA-JEPA base; world objective disabled",
    },
    "jepa_wm20k": {
        "steps": 20_000,
        "batch_size": 16,
        "sample_updates": 320_000,
        "cameras": ["top", "wrist"],
        "state_dim": 6,
        "action_dim": 6,
        "action_norm": "MIN_MAX",
        "chunk_size": 7,
        "frozen": "V-JEPA target encoder frozen; Qwen trainable",
        "pretrained": "LIBERO VLA-JEPA base; auxiliary latent loss weight 0.1",
    },
    "fastwam": {
        "steps": "20K + 8K",
        "batch_size": 16,
        "sample_updates": 448_000,
        "cameras": ["top", "wrist"],
        "state_dim": "6 padded to 8",
        "action_dim": "6 padded to 7",
        "action_norm": "MEAN_STD",
        "chunk_size": 32,
        "frozen": "Wan video expert frozen except last 4 MoT layers in final 8K",
        "pretrained": "LIBERO FastWAM action path; Wan2.2 video backbone",
    },
}


def wilson_interval(successes: int, episodes: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / episodes
    denom = 1 + z**2 / episodes
    center = (p + z**2 / (2 * episodes)) / denom
    margin = z * math.sqrt(p * (1 - p) / episodes + z**2 / (4 * episodes**2)) / denom
    return center - margin, center + margin


def load_yaml_protocol(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text())
    robot = raw["robot"]
    return {
        "config": str(path.relative_to(REPO)),
        "spawn": robot["reset_cube"],
        "reset_arm": robot["reset_arm"],
        "home_joints": robot.get("home_joints"),
        "x_range": robot.get("cube_random_x_range"),
        "y_range": robot.get("cube_random_y_range"),
        "yaw_range": robot.get("cube_random_yaw_range"),
        "cameras_rendered": robot["camera_names"],
        "cameras_used": len(raw.get("rename_map") or {}) or len(robot["camera_names"]),
        "inference": raw["inference"]["type"],
        "fps": raw["fps"],
        "episode_time_s": raw["eval"]["episode_time_s"],
    }


def load_checkpoint_configs() -> dict[str, Any]:
    out = {}
    for key, rel in CHECKPOINTS.items():
        path = REPO / rel / "config.json"
        if not path.exists():
            continue
        cfg = json.loads(path.read_text())
        out[key] = {
            "path": rel,
            "type": cfg.get("type"),
            "state_dim": cfg.get("state_dim", cfg.get("proprio_dim")),
            "action_dim": cfg.get("action_dim"),
            "chunk_size": cfg.get("chunk_size", cfg.get("action_horizon")),
            "n_action_steps": cfg.get("n_action_steps"),
            "normalization": cfg.get("normalization_mapping"),
            "freeze_qwen": cfg.get("freeze_qwen"),
            "enable_world_model": cfg.get("enable_world_model"),
            "world_model_loss_weight": cfg.get("world_model_loss_weight"),
            "freeze_video_expert": cfg.get("freeze_video_expert"),
            "loss": cfg.get("loss"),
            "input_features": {
                name: value.get("shape") for name, value in (cfg.get("input_features") or {}).items()
            },
        }
    return out


def _fixed_list_column(table, name: str) -> np.ndarray:
    values = table[name].combine_chunks()
    return np.asarray(values.values).reshape(len(values), values.type.list_size)


def dataset_analysis() -> dict[str, Any]:
    files = sorted((DATASET / "data").glob("**/*.parquet"))
    tables = [
        pq.read_table(path, columns=["action", "observation.state", "episode_index", "frame_index"])
        for path in files
    ]
    import pyarrow as pa

    table = pa.concat_tables(tables)
    action = _fixed_list_column(table, "action")
    state = _fixed_list_column(table, "observation.state")
    episodes = np.asarray(table["episode_index"].combine_chunks())
    frames = np.asarray(table["frame_index"].combine_chunks())
    first = frames == 0
    counts = Counter(int(v) for v in episodes)
    lengths = np.asarray(list(counts.values()))
    grip = action[:, -1]
    # Lab 01's continuous leader sweep spends most time in two modes. The
    # middle region quantifies ambiguous supervision for regression/flow heads.
    grip_regions = {
        "closed_lt_0.15": float(np.mean(grip < 0.15)),
        "transition_0.15_to_0.75": float(np.mean((grip >= 0.15) & (grip <= 0.75))),
        "open_gt_0.75": float(np.mean(grip > 0.75)),
    }
    closed_value = float(np.median(grip[grip < 0.15]))
    open_value = float(np.median(grip[grip > 0.75]))
    binary_grip = np.where(grip < 0.45, closed_value, open_value)
    gripper_events = []
    action_velocity = []
    action_curvature = []
    for episode in sorted(counts):
        episode_action = action[episodes == episode]
        episode_grip = episode_action[:, -1]
        action_velocity.append(np.diff(episode_action, axis=0))
        action_curvature.append(np.diff(episode_action, n=2, axis=0))
        open_idx = np.flatnonzero(episode_grip > 0.75)
        first_open = int(open_idx[0]) if len(open_idx) else None
        close_idx = (
            np.flatnonzero(episode_grip[first_open + 1 :] < 0.15)
            if first_open is not None
            else np.asarray([])
        )
        first_close = (
            first_open + 1 + int(close_idx[0])
            if first_open is not None and len(close_idx)
            else None
        )
        reopen_idx = (
            np.flatnonzero(episode_grip[first_close + 1 :] > 0.75)
            if first_close is not None
            else np.asarray([])
        )
        first_reopen = (
            first_close + 1 + int(reopen_idx[0])
            if first_close is not None and len(reopen_idx)
            else None
        )
        ramp_start_idx = (
            np.flatnonzero(episode_grip[first_open:first_close] < 0.75)
            if first_open is not None and first_close is not None
            else np.asarray([])
        )
        ramp_frames = (
            first_close - (first_open + int(ramp_start_idx[0]))
            if first_close is not None and len(ramp_start_idx)
            else None
        )
        gripper_events.append(
            {
                "episode": episode,
                "open_frame": first_open,
                "close_frame": first_close,
                "reopen_frame": first_reopen,
                "close_ramp_frames": ramp_frames,
            }
        )

    def _event_summary(key: str) -> dict[str, Any]:
        values = [event[key] for event in gripper_events if event[key] is not None]
        return {
            "episodes": len(values),
            "median_frames": float(np.median(values)) if values else None,
            "median_s": float(np.median(values) / 20.0) if values else None,
        }

    velocity = np.concatenate(action_velocity)
    curvature = np.concatenate(action_curvature)
    return {
        "frames": int(len(action)),
        "episodes": int(len(counts)),
        "episode_length_frames": {
            "min": int(lengths.min()),
            "median": float(np.median(lengths)),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "start_joint_pos": {
            "mean": state[first, :6].mean(axis=0).tolist(),
            "std": state[first, :6].std(axis=0).tolist(),
            "max_std": float(state[first, :6].std(axis=0).max()),
        },
        "gripper": {
            "quantiles": {
                str(q): float(np.quantile(grip, q)) for q in (0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99)
            },
            "regions": grip_regions,
            "binary_approximation": {
                "threshold_rad": 0.45,
                "closed_value_rad": closed_value,
                "open_value_rad": open_value,
                "rmse_rad": float(np.sqrt(np.mean((binary_grip - grip) ** 2))),
                "current_vla_jepa_bc_uses_binarization": False,
            },
            "events": {
                "first_open": _event_summary("open_frame"),
                "first_close_after_open": _event_summary("close_frame"),
                "first_reopen_after_close": _event_summary("reopen_frame"),
                "close_ramp": _event_summary("close_ramp_frames"),
            },
        },
        "joint_action_std": action.std(axis=0).tolist(),
        "trajectory": {
            "joint_delta_rms_per_frame": np.sqrt(np.mean(velocity**2, axis=0)).tolist(),
            "joint_curvature_rms_per_frame2": np.sqrt(
                np.mean(curvature**2, axis=0)
            ).tolist(),
        },
    }


def parse_eval_logs() -> dict[str, Any]:
    out = {}
    for path in sorted((REPO / "outputs/eval").glob("*.log")):
        text = path.read_text(errors="replace")
        matches = re.findall(r"Eval complete: (\d+)/(\d+) success \((\d+)%\)", text)
        if not matches:
            continue
        successes, episodes, rate = map(int, matches[-1])
        out[path.name] = {
            "successes": successes,
            "episodes": episodes,
            "rate_pct": rate,
        }
        funnel_lines = re.findall(r"Stage funnel \(\d+ ep\): (.+)", text)
        if funnel_lines:
            out[path.name]["funnel"] = {
                key: {"count": int(count), "episodes": int(total)}
                for key, count, total in re.findall(
                    r"([a-z0-9_]+)=(\d+)/(\d+)", funnel_lines[-1]
                )
            }
    return out


def load_probes() -> dict[str, Any]:
    out = {}
    for key, rel in PROBES.items():
        path = REPO / rel
        if path.exists():
            out[key] = json.loads(path.read_text())
    return out


def load_generated_diagnostics(prefix: str) -> dict[str, Any]:
    out = {}
    for path in sorted((REPO / ".tmp").glob(f"{prefix}_*.json")):
        out[path.stem.removeprefix(f"{prefix}_")] = json.loads(path.read_text())
    return out


def build_bundle() -> dict[str, Any]:
    runs = []
    for run in REFERENCE_RUNS:
        row = dict(run)
        lo, hi = wilson_interval(row["successes"], row["episodes"])
        row["success_rate"] = row["successes"] / row["episodes"]
        row["wilson95"] = [lo, hi]
        # Only these pairs hold all major protocol fields constant.
        row["comparison_group"] = {
            "act15_full": "act_state_ablation",
            "act6_full": "act_state_ablation",
            "molmo15_full": "molmo_state_ablation",
            "molmo6_full": "molmo_state_ablation",
        }.get(row["id"], "protocol_confounded")
        runs.append(row)

    protocol_files = [
        "rollout_act.yaml",
        "rollout_smolvla.yaml",
        "rollout_molmoact2.yaml",
        "rollout_fastwam.yaml",
        "rollout_vla_jepa_demo_start.yaml",
        "rollout_vla_jepa_demo_fixed.yaml",
    ]
    return {
        "schema_version": 1,
        "dataset": dataset_analysis(),
        "reference_runs": runs,
        "training_protocols": TRAINING_PROTOCOLS,
        "protocols": {
            name.removesuffix(".yaml"): load_yaml_protocol(LAB / "configs" / name)
            for name in protocol_files
        },
        "checkpoints": load_checkpoint_configs(),
        "eval_logs": parse_eval_logs(),
        "visual_probes": load_probes(),
        "controlled_grounding": load_generated_diagnostics("grounding"),
        "open_loop": load_generated_diagnostics("openloop"),
        "world_model_semantics": {
            "vla_jepa_bc10k": (
                "enable_world_model=false; inference uses current-frame Qwen embodied-action "
                "tokens and action DiT only"
            ),
            "vla_jepa_wm20k": (
                "auxiliary frozen-V-JEPA latent predictor at weight 0.1; predictor is unused "
                "during action inference"
            ),
            "fastwam": (
                "freeze_video_expert=true and lambda_video=0; action inference uses a "
                "first-frame video KV cache, not imagined future frames"
            ),
            "paper_scale_contrast": {
                "fastwam": (
                    "The paper attributes most WAM benefit to video co-training; its LIBERO "
                    "protocol uses 500 demos across 10 tasks and 20K steps. Lab 01 removes "
                    "video co-training and has 50 demos for one new embodiment."
                ),
                "vla_jepa": (
                    "The paper pretrains on 220K human videos plus 76K robot trajectories, "
                    "then fine-tunes on about 2K LIBERO demos. Lab 01 directly adapts the "
                    "released LIBERO model to 50 SO-101 demos."
                ),
                "sources": [
                    "https://arxiv.org/abs/2603.16666",
                    "https://arxiv.org/abs/2602.10098",
                ],
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / ".tmp/model_gap_analysis.json",
        help="JSON evidence bundle path",
    )
    args = parser.parse_args()
    bundle = build_bundle()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"Wrote {args.output}")
    print(
        f"Dataset: {bundle['dataset']['episodes']} episodes, "
        f"{bundle['dataset']['frames']} frames; "
        f"{len(bundle['reference_runs'])} reference runs; "
        f"{len(bundle['controlled_grounding'])} controlled grounding probes; "
        f"{len(bundle['open_loop'])} open-loop probes"
    )


if __name__ == "__main__":
    main()
