#!/usr/bin/env python3
"""Controlled cube-grounding probe for Lab 01 policies.

MuJoCo renders a counterfactual grid where arm pose, proprio, lighting, and
background are fixed and only cube XY changes. A matched no-cube render (cube
temporarily moved below the table) isolates response to the cube itself.

The probe is policy-agnostic for the local ACT, SmolVLA, MolmoAct2, VLA-JEPA,
and FastWAM checkpoints. It reports raw-joint action sensitivity plus a
kinematic question: does any target in the predicted chunk bring the gripper
near the cube?
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco as mj
import numpy as np
import pyarrow.parquet as pq
import torch
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.policies.factory import get_policy_class, make_pre_post_processors  # noqa: E402
from lerobot.processor.rename_processor import rename_stats  # noqa: E402
from scipy.stats import spearmanr

from simstudio.robots.so101_mujoco.configuration_so101_mujoco import (  # noqa: E402
    SO101MujocoConfig,
)
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "datasets/so101-simstudio-lab01-pnp"
DEMO_MEAN_HOME = [-0.180, -1.642, 1.577, 1.081, 0.079, -0.045]
GRID_X = [0.265, 0.300, 0.335]
GRID_Y = [0.165, 0.200, 0.235]


def _preserve_vla_jepa_dims() -> None:
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    orig = VLAJEPAConfig.validate_features

    def wrapped(self) -> None:
        saved_state = int(self.state_dim)
        saved_action = int(self.action_dim)
        orig(self)
        if saved_state != self.state_dim:
            self.state_dim = saved_state
            feature = (self.input_features or {}).get("observation.state")
            if feature is not None:
                self.input_features["observation.state"] = replace(
                    feature, shape=(saved_state,)
                )
        if saved_action and saved_action != self.action_dim:
            self.action_dim = saved_action

    VLAJEPAConfig.validate_features = wrapped  # type: ignore[method-assign]


def _rename_map(policy_type: str) -> dict[str, str]:
    if policy_type == "smolvla":
        return {
            "observation.images.camera_top": "observation.images.camera1",
            "observation.images.camera_front": "observation.images.camera2",
            "observation.images.camera_wrist": "observation.images.camera3",
        }
    if policy_type == "molmoact2":
        return {
            "observation.images.camera_top": "observation.images.cam0",
            "observation.images.camera_wrist": "observation.images.cam1",
        }
    if policy_type in {"vla_jepa", "fastwam"}:
        return {
            "observation.images.camera_top": "observation.images.image",
            "observation.images.camera_wrist": "observation.images.image2",
        }
    return {}


def _normalization_name(cfg: Any) -> str:
    raw = (getattr(cfg, "normalization_mapping", None) or {}).get("ACTION", "MEAN_STD")
    return getattr(raw, "name", str(raw))


def _unnormalize(pred: np.ndarray, stats: dict[str, Any], mode: str) -> np.ndarray:
    mode = mode.upper()
    if "QUANTILE10" in mode:
        low, high = stats["q10"], stats["q90"]
    elif "QUANTILES" in mode:
        low, high = stats["q01"], stats["q99"]
    elif "MIN_MAX" in mode:
        low, high = stats["min"], stats["max"]
    elif "IDENTITY" in mode:
        return pred
    else:
        mean = np.asarray(stats["mean"], dtype=np.float64)
        std = np.asarray(stats["std"], dtype=np.float64)
        return pred * std + mean
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    return 0.5 * (pred + 1.0) * (high - low) + low


def _dataset_stats() -> dict[str, Any]:
    return json.loads((DATASET / "meta/stats.json").read_text())


def _patch_processor_state_stats(preprocessor: Any, state_stats: dict[str, Any]) -> None:
    """Match eval.py: checkpoints may retain 15-D stats for a 6-D policy."""
    for step in preprocessor.steps:
        stats = getattr(step, "stats", None)
        if isinstance(stats, dict) and "observation.state" in stats:
            old = stats["observation.state"]
            stats["observation.state"] = {
                key: (
                    torch.as_tensor(value, dtype=old[key].dtype, device=old[key].device)
                    if isinstance(old.get(key), torch.Tensor)
                    else value
                )
                for key, value in state_stats.items()
            }


def _to_image_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0


def _robot_observation(robot: SO101MujocoRobot) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    obs = robot.get_observation()
    images = {
        f"observation.images.camera_{name}": _to_image_tensor(obs[f"camera_{name}"])
        for name in robot.config.camera_names
    }
    state = torch.tensor(
        [float(obs[f"{name}.pos"]) for name in robot.JOINT_NAMES], dtype=torch.float32
    )
    return images, state


def _render_counterfactuals(
    robot: SO101MujocoRobot,
) -> tuple[list[dict[str, torch.Tensor]], list[dict[str, torch.Tensor]], torch.Tensor]:
    present, absent = [], []
    fixed_state = None
    block_geom_id = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_GEOM, "block")
    block_rgba = robot.model.geom_rgba[block_geom_id].copy()
    for x in GRID_X:
        for y in GRID_Y:
            robot.reset_to_home_position()
            robot.set_block_position_direct(x, y, 0.0125)
            images, state = _robot_observation(robot)
            present.append(images)
            if fixed_state is None:
                fixed_state = state

            # Preserve the body pose because a scene light tracks the block.
            # Alpha=0 removes only the cube pixels without changing lighting.
            robot.model.geom_rgba[block_geom_id, 3] = 0.0
            images_no_cube, _ = _robot_observation(robot)
            robot.model.geom_rgba[block_geom_id] = block_rgba
            absent.append(images_no_cube)
    assert fixed_state is not None
    return present, absent, fixed_state


def _variant_images(
    variant: str,
    present: dict[str, torch.Tensor],
    absent: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if variant == "normal":
        return present
    if variant == "no_cube":
        return absent
    keep = {
        "top_only": "camera_top",
        "wrist_only": "camera_wrist",
        "front_only": "camera_front",
    }[variant]
    return {
        key: value if key.endswith(keep) else absent[key]
        for key, value in present.items()
    }


def _cube_pixel_summary(images: list[dict[str, torch.Tensor]]) -> dict[str, Any]:
    summary = {}
    for key in images[0]:
        areas = []
        for scene in images:
            rgb = scene[key]
            green = (rgb[1] > 0.45) & (rgb[1] > 1.45 * rgb[0]) & (rgb[1] > 1.25 * rgb[2])
            areas.append(int(green.sum()))
        summary[key.rsplit(".", 1)[-1]] = {
            "area_px_mean": float(np.mean(areas)),
            "area_px_min": int(np.min(areas)),
            "area_px_max": int(np.max(areas)),
            "equivalent_width_px_mean": float(np.mean(np.sqrt(areas))),
        }
    return summary


def _dataset_start_states(limit: int) -> np.ndarray:
    rows = []
    for path in sorted((DATASET / "data").glob("**/*.parquet")):
        table = pq.read_table(path, columns=["observation.state", "frame_index"])
        values = table["observation.state"].combine_chunks()
        state = np.asarray(values.values).reshape(len(values), values.type.list_size)
        frame = np.asarray(table["frame_index"].combine_chunks())
        rows.extend(state[frame == 0, :6])
        if len(rows) >= limit:
            break
    return np.asarray(rows[:limit], dtype=np.float32)


def _pairwise_rank_correlation(actions: np.ndarray, xy: np.ndarray) -> float:
    action_dist, xy_dist = [], []
    for i, j in combinations(range(len(xy)), 2):
        action_dist.append(float(np.linalg.norm(actions[i] - actions[j])))
        xy_dist.append(float(np.linalg.norm(xy[i] - xy[j])))
    result = spearmanr(xy_dist, action_dist)
    return float(result.statistic) if np.isfinite(result.statistic) else 0.0


def _linear_slopes(actions: np.ndarray, xy: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(xy)), xy - xy.mean(axis=0)])
    return np.linalg.lstsq(design, actions.reshape(len(xy), -1), rcond=None)[0][1:]


def _fk_chunk_metrics(
    model: mj.MjModel,
    data: mj.MjData,
    qpos_ids: list[int],
    chunk: np.ndarray,
    cube_xyz: np.ndarray,
    site_id: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    qpos_saved = data.qpos.copy()
    best = float("inf")
    best_xyz = np.zeros(3)
    endpoint_xyz = np.zeros(3)
    try:
        for action in chunk:
            for qpos_id, value in zip(qpos_ids, action, strict=False):
                data.qpos[qpos_id] = float(value)
            data.qvel[:] = 0
            mj.mj_forward(model, data)
            endpoint_xyz = data.site_xpos[site_id].copy()
            distance = float(np.linalg.norm(endpoint_xyz - cube_xyz))
            if distance < best:
                best = distance
                best_xyz = endpoint_xyz.copy()
    finally:
        data.qpos[:] = qpos_saved
        data.qvel[:] = 0
        mj.mj_forward(model, data)
    return best, best_xyz, endpoint_xyz


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config_json = json.loads((args.checkpoint / "config.json").read_text())
    policy_type = config_json["type"]
    if policy_type == "vla_jepa":
        _preserve_vla_jepa_dims()
    if policy_type == "fastwam":
        from simstudio.policies.fastwam_pad import patch_fastwam_pad_action_channels

        patch_fastwam_pad_action_channels(6)

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    cfg.pretrained_path = str(args.checkpoint)
    cfg.device = "cuda"
    if policy_type == "molmoact2":
        # CUDA/HIP graph capture segfaults on gfx1151; same guard as eval.cmd.
        cfg.enable_inference_cuda_graph = False
    rename = _rename_map(policy_type)
    stats = _dataset_stats()
    renamed_stats = rename_stats(stats, rename)
    renamed_stats["observation.state"] = {
        key: value[:6] for key, value in renamed_stats["observation.state"].items()
    }

    robot = SO101MujocoRobot(
        SO101MujocoConfig(
            id="model-gap-probe",
            action_mode="position",
            reset_mode="auto",
            reset_arm="home",
            reset_cube="none",
            home_joints=DEMO_MEAN_HOME,
            camera_names=["front", "top", "wrist"],
            camera_width=640,
            camera_height=480,
            render_window=False,
        )
    )
    robot.connect()
    # The scene's second light targets the block body. Disable tracked lights so
    # cube XY cannot leak through global illumination in the no-cube control.
    robot.model.light_active[robot.model.light_targetbodyid >= 0] = False
    present, absent, fixed_state = _render_counterfactuals(robot)
    cube_pixels = _cube_pixel_summary(present)
    robot.disconnect()

    # Keep EGL/ROCm out of the same live context for large VLMs (MolmoAct2 can
    # otherwise trigger allocator corruption on integrated AMD GPUs).
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

    xy = np.asarray([(x, y) for x in GRID_X for y in GRID_Y], dtype=np.float64)
    image_keys = list(cfg.image_features)
    src_by_dst = {dst: src for src, dst in rename.items()}
    action_mode = _normalization_name(cfg)

    def predict(
        images: dict[str, torch.Tensor], state: torch.Tensor = fixed_state
    ) -> np.ndarray:
        batch: dict[str, Any] = {
            "observation.state": state.unsqueeze(0),
            "task": "Pick up the cube and place it in the box.",
        }
        for dst in image_keys:
            src = src_by_dst.get(dst, dst)
            if src not in images:
                # SmolVLA declares empty_camera_0 in config and synthesizes it
                # internally when absent.
                continue
            batch[src] = images[src].unsqueeze(0)
        policy.reset()
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with torch.no_grad():
            chunk = policy.predict_action_chunk(preprocessor(batch))[0]
        raw = _unnormalize(
            chunk.float().cpu().numpy().astype(np.float64), stats["action"], action_mode
        )
        return raw[:, :6]

    variants = ["normal", "no_cube", "top_only", "wrist_only"]
    if "observation.images.camera_front" in src_by_dst.values() or not rename:
        variants.append("front_only")

    result_variants: dict[str, Any] = {}
    variant_chunks: dict[str, np.ndarray] = {}
    normal_chunks = None
    for variant in variants:
        chunks = np.stack(
            [predict(_variant_images(variant, p, a)) for p, a in zip(present, absent, strict=True)]
        )
        variant_chunks[variant] = chunks
        if variant == "normal":
            normal_chunks = chunks
        centered = chunks - chunks.mean(axis=0, keepdims=True)
        flat_centered = centered.reshape(len(xy), -1)
        slopes = _linear_slopes(chunks, xy)
        spread = float(np.sqrt(np.mean(centered**2)))
        result_variants[variant] = {
            "action_spread_rad": spread,
            "pairwise_xy_action_spearman": (
                _pairwise_rank_correlation(chunks.reshape(len(xy), -1), xy)
                if spread >= 1e-4
                else 0.0
            ),
            "slope_norm_per_m": {
                "x": float(np.linalg.norm(slopes[0])),
                "y": float(np.linalg.norm(slopes[1])),
            },
            "gripper_spread_rad": float(np.sqrt(np.mean(centered[..., -1] ** 2))),
            "rank": int(np.linalg.matrix_rank(flat_centered)),
        }

    assert normal_chunks is not None
    no_cube_chunks = variant_chunks["no_cube"]
    start_states = _dataset_start_states(len(xy))
    proprio_chunks = np.stack(
        [predict(present[len(present) // 2], torch.from_numpy(state)) for state in start_states]
    )
    proprio_centered = proprio_chunks - proprio_chunks.mean(axis=0, keepdims=True)
    cube_effect = normal_chunks - no_cube_chunks
    cube_position_effect = cube_effect - cube_effect.mean(axis=0, keepdims=True)
    fk_model = mj.MjModel.from_xml_path(
        str(REPO / "SO101/scenes/simple_pick/scene.xml")
    )
    fk_data = mj.MjData(fk_model)
    qpos_ids = [
        int(fk_model.jnt_qposadr[mj.mj_name2id(fk_model, mj.mjtObj.mjOBJ_JOINT, name)])
        for name in SO101MujocoRobot.JOINT_NAMES
    ]
    gripper_site = mj.mj_name2id(fk_model, mj.mjtObj.mjOBJ_SITE, "gripperframe")
    fk_metrics = [
        _fk_chunk_metrics(
            fk_model,
            fk_data,
            qpos_ids,
            chunk,
            np.asarray([point[0], point[1], 0.015], dtype=np.float64),
            gripper_site,
        )
        for chunk, point in zip(normal_chunks, xy, strict=True)
    ]
    fk_dist = np.asarray([metric[0] for metric in fk_metrics])
    fk_closest_xy = np.asarray([metric[1][:2] for metric in fk_metrics])
    fk_endpoint_xy = np.asarray([metric[2][:2] for metric in fk_metrics])

    def _axis_corr(predicted_xy: np.ndarray) -> dict[str, float]:
        result = {}
        for i, axis in enumerate(("x", "y")):
            result[axis] = (
                float(np.corrcoef(predicted_xy[:, i], xy[:, i])[0, 1])
                if np.std(predicted_xy[:, i]) > 1e-9
                else 0.0
            )
        return result

    result = {
        "policy": policy_type,
        "checkpoint": str(args.checkpoint),
        "protocol": {
            "grid_x_m": GRID_X,
            "grid_y_m": GRID_Y,
            "arm_home_rad": DEMO_MEAN_HOME,
            "fixed_proprio": fixed_state.tolist(),
            "seed": args.seed,
            "action_normalization": action_mode,
            "counterfactual": (
                "cube geom alpha=0 with block-targeted light disabled; "
                "arm, proprio, cameras, and background fixed"
            ),
        },
        "cube_pixels": cube_pixels,
        "variants": result_variants,
        "cube_presence_effect_rms_rad": float(np.sqrt(np.mean(cube_effect**2))),
        "cube_position_effect_rms_rad": float(np.sqrt(np.mean(cube_position_effect**2))),
        "cube_position_effect_to_normal_spread": float(
            np.sqrt(np.mean(cube_position_effect**2))
            / (result_variants["normal"]["action_spread_rad"] + 1e-12)
        ),
        "fixed_image_proprio_intervention": {
            "start_state_spread_rad": float(
                np.sqrt(np.mean((start_states - start_states.mean(axis=0)) ** 2))
            ),
            "action_spread_rad": float(np.sqrt(np.mean(proprio_centered**2))),
            "note": (
                "fixed center-cube image with nine recorded episode-start proprio vectors; "
                "off-manifold intervention measures dependence, not task accuracy"
            ),
        },
        "fk_min_gripper_cube_distance_m": {
            "mean": float(np.mean(fk_dist)),
            "min": float(np.min(fk_dist)),
            "max": float(np.max(fk_dist)),
            "within_4cm_fraction": float(np.mean(np.asarray(fk_dist) <= 0.04)),
            "closest_xy_rmse_m": float(np.sqrt(np.mean((fk_closest_xy - xy) ** 2))),
            "closest_xy_axis_correlation": _axis_corr(fk_closest_xy),
            "endpoint_xy_axis_correlation": _axis_corr(fk_endpoint_xy),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
