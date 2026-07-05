"""Validate a recorded LeRobot dataset for quality and integrity.

Checks:
  - Metadata consistency (info.json counters vs actual data)
  - Frame rate (timestamp intervals match fps)
  - Action range (within MuJoCo joint limits)
  - Completeness (episodes, videos, parquet files exist)
  - Data integrity (parquet readable, no corruption)

Usage:
    uv run python -m simstudio.scripts.validate_dataset \
        --root ./datasets/leader-test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

# MuJoCo SO-101 joint limits (radians) from actuator_ctrlrange
MUJOCO_JOINT_LIMITS = {
    "shoulder_pan": (-1.9198621771937326, 1.9198621771937326),
    "shoulder_lift": (-1.7453292519943295, 1.7453292519943295),
    "elbow_flex": (-1.6929693744328523, 1.6929693744328523),
    "wrist_flex": (-1.6580628494556928, 1.6580627293335335),
    "wrist_roll": (-2.7438472969992493, 2.841206309382605),
    "gripper": (-0.17453297762778586, 1.7453291995659765),
}

# Expected action keys for SO-101 position mode
EXPECTED_ACTION_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


class ValidationResult:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.info: list[str] = []

    def error(self, msg: str):
        self.errors.append(msg)

    def warn(self, msg: str):
        self.warnings.append(msg)

    def ok(self, msg: str):
        self.info.append(msg)

    @property
    def passed(self) -> bool:
        return len(self.errors) == 0

    def summary(self) -> str:
        lines = []
        if self.info:
            lines.append("=== INFO ===")
            for m in self.info:
                lines.append(f"  {m}")
        if self.warnings:
            lines.append("=== WARNINGS ===")
            for m in self.warnings:
                lines.append(f"  ! {m}")
        if self.errors:
            lines.append("=== ERRORS ===")
            for m in self.errors:
                lines.append(f"  X {m}")
        lines.append("")
        if self.passed:
            lines.append("PASSED")
        else:
            lines.append(f"FAILED ({len(self.errors)} errors, {len(self.warnings)} warnings)")
        return "\n".join(lines)


def validate_dataset(root: Path) -> ValidationResult:
    result = ValidationResult()

    # --- 1. Check meta/info.json ---
    info_path = root / "meta" / "info.json"
    if not info_path.exists():
        result.error(f"meta/info.json not found at {info_path}")
        return result

    with open(info_path) as f:
        info = json.load(f)

    result.ok(f"Dataset: {info.get('repo_id', 'unknown')}")
    result.ok(f"Robot type: {info.get('robot_type', 'unknown')}")
    result.ok(f"FPS: {info['fps']}")
    result.ok(f"Version: {info.get('codebase_version', 'unknown')}")

    expected_episodes = info["total_episodes"]
    expected_frames = info["total_frames"]
    result.ok(f"Episodes: {expected_episodes}, Frames: {expected_frames}")

    # --- 2. Check parquet files exist and count frames ---
    data_dir = root / "data"
    if not data_dir.exists():
        result.error(f"data/ directory not found at {data_dir}")
        return result

    parquet_files = sorted(data_dir.rglob("*.parquet"))
    if not parquet_files:
        result.error("No parquet files found in data/")
        return result

    result.ok(f"Found {len(parquet_files)} parquet file(s)")

    # Read all parquet data
    all_frames = []
    for pq_path in parquet_files:
        try:
            table = pq.read_table(pq_path)
            all_frames.append(table)
        except Exception as e:
            result.error(f"Failed to read {pq_path.name}: {e}")

    if not all_frames:
        result.error("No readable parquet data")
        return result

    combined = all_frames[0]
    for t in all_frames[1:]:
        combined = combined.append(t)

    actual_frames = combined.num_rows
    result.ok(f"Actual frames in parquet: {actual_frames}")

    if actual_frames != expected_frames:
        result.error(f"Frame count mismatch: info.json says {expected_frames}, parquet has {actual_frames}")

    # --- 3. Check episode count and completeness ---
    episode_indices = combined.column("episode_index").to_pylist()
    unique_episodes = sorted(set(episode_indices))
    actual_episodes = len(unique_episodes)

    result.ok(f"Actual episodes: {actual_episodes}")

    if actual_episodes != expected_episodes:
        result.error(f"Episode count mismatch: info.json says {expected_episodes}, found {actual_episodes}")

    # Check episode indices are contiguous
    expected_indices = list(range(expected_episodes))
    if unique_episodes != expected_indices:
        missing = set(expected_indices) - set(unique_episodes)
        extra = set(unique_episodes) - set(expected_indices)
        if missing:
            result.error(f"Missing episodes: {sorted(missing)}")
        if extra:
            result.error(f"Unexpected episode indices: {sorted(extra)}")

    # --- 4. Check frame rate (timestamp intervals) ---
    fps = info["fps"]
    expected_dt = 1.0 / fps
    tolerance = expected_dt * 0.1  # 10% tolerance

    frame_indices = combined.column("frame_index").to_pylist()
    timestamps = combined.column("timestamp").to_pylist()

    # Group by episode and check timestamps
    episode_data: dict[int, list[tuple[int, float]]] = {}
    for ep_idx, frame_idx, ts in zip(episode_indices, frame_indices, timestamps, strict=True):
        if ep_idx not in episode_data:
            episode_data[ep_idx] = []
        episode_data[ep_idx].append((frame_idx, ts))

    frame_rate_errors = 0
    for ep_idx in sorted(episode_data.keys()):
        frames = sorted(episode_data[ep_idx], key=lambda x: x[0])

        # Check frame indices are 0-based and contiguous
        frame_ids = [f[0] for f in frames]
        if frame_ids != list(range(len(frame_ids))):
            result.warn(f"Episode {ep_idx}: non-contiguous frame indices")
            frame_rate_errors += 1

        # Check timestamp intervals
        ts_values = [f[1] for f in frames]
        for i in range(1, len(ts_values)):
            dt = ts_values[i] - ts_values[i - 1]
            if abs(dt - expected_dt) > tolerance:
                frame_rate_errors += 1
                if frame_rate_errors <= 3:  # Only show first 3 errors
                    result.warn(f"Episode {ep_idx}, frame {i}: dt={dt:.4f}s, expected {expected_dt:.4f}s")

    if frame_rate_errors == 0:
        result.ok("Frame rate consistent across all episodes")
    elif frame_rate_errors > 3:
        result.warn(f"Frame rate issues in {frame_rate_errors} frames (showing first 3)")

    # --- 5. Check action range ---
    action_col = combined.column("action").to_pylist()
    all_actions = np.array(action_col)

    if all_actions.ndim == 2 and all_actions.shape[1] == len(EXPECTED_ACTION_KEYS):
        for i, (key, (lo, hi)) in enumerate(
            zip(EXPECTED_ACTION_KEYS, MUJOCO_JOINT_LIMITS.values(), strict=True)
        ):
            col = all_actions[:, i]
            min_val = float(np.min(col))
            max_val = float(np.max(col))

            # Allow 5% tolerance beyond limits
            margin = (hi - lo) * 0.05
            if min_val < lo - margin:
                result.warn(f"Action {key}: min={min_val:.4f} below limit {lo:.4f}")
            if max_val > hi + margin:
                result.warn(f"Action {key}: max={max_val:.4f} above limit {hi:.4f}")

        # Check for NaN/Inf
        nan_count = int(np.isnan(all_actions).sum())
        inf_count = int(np.isinf(all_actions).sum())
        if nan_count > 0:
            result.error(f"Actions contain {nan_count} NaN values")
        if inf_count > 0:
            result.error(f"Actions contain {inf_count} Inf values")
        if nan_count == 0 and inf_count == 0:
            result.ok("Actions clean (no NaN/Inf)")
    else:
        result.warn(f"Unexpected action shape: {all_actions.shape}")

    # --- 6. Check videos exist ---
    videos_dir = root / "videos"
    if videos_dir.exists():
        video_keys = info.get("features", {})
        video_features = {k: v for k, v in video_keys.items() if v.get("dtype") == "video"}

        for feat_name in video_features:
            feat_dir = videos_dir / feat_name
            if feat_dir.exists():
                video_files = list(feat_dir.rglob("*.mp4"))
                result.ok(f"Video {feat_name}: {len(video_files)} file(s)")
            else:
                result.warn(f"Video directory not found: {feat_dir}")
    else:
        result.warn("No videos/ directory found")

    # --- 7. Check stats.json ---
    stats_path = root / "meta" / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        result.ok(f"stats.json found with {len(stats)} feature(s)")
    else:
        result.warn("meta/stats.json not found")

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate LeRobot dataset")
    parser.add_argument(
        "--root",
        required=True,
        help="Dataset root directory (contains meta/ and data/)",
    )
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        print(f"Error: Directory not found: {root}")
        sys.exit(1)

    print(f"Validating dataset at: {root}\n")
    result = validate_dataset(root)
    print(result.summary())
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
