#!/usr/bin/env python3
"""Phase 0: quality proxies on Lab 01 teleop parquet (no cube-coverage — spawn is not stored)."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO / "datasets/so101-simstudio-lab01-pnp"
FPS = 20.0


def _list_col(table, name: str) -> np.ndarray:
    values = table[name].combine_chunks()
    return np.asarray(values.values).reshape(len(values), values.type.list_size)


def analyze(root: Path) -> dict:
    files = sorted((root / "data").glob("**/*.parquet"))
    if not files:
        raise SystemExit(f"No parquet under {root}/data")
    table = pa.concat_tables(
        [
            pq.read_table(p, columns=["action", "observation.state", "episode_index"])
            for p in files
        ]
    )
    action = _list_col(table, "action")
    state = _list_col(table, "observation.state")
    episodes = np.asarray(table["episode_index"].combine_chunks())
    counts = Counter(int(v) for v in episodes)
    ee = state[:, 12:15]
    lengths, path_len, jerk, n_close = [], [], [], []
    for ep in sorted(counts):
        mask = episodes == ep
        xyz = ee[mask]
        delta = np.diff(xyz, axis=0)
        path_len.append(float(np.sum(np.linalg.norm(delta, axis=1))))
        acc = np.diff(delta, axis=0)
        jerk.append(float(np.sum(np.linalg.norm(acc, axis=1))))
        grip = action[mask, -1]
        edges = np.diff((grip < 0.15).astype(np.int8))
        n_close.append(int(np.sum(edges == 1)))
        lengths.append(int(counts[ep]))
    return {
        "root": str(root),
        "episodes": int(len(counts)),
        "frames": int(len(action)),
        "duration_s": {
            "min": min(lengths) / FPS,
            "median": float(np.median(lengths) / FPS),
            "max": max(lengths) / FPS,
        },
        "ee_path_m": {
            "median": float(np.median(path_len)),
            "p90": float(np.percentile(path_len, 90)),
            "max": float(max(path_len)),
        },
        "ee_jerk_proxy": {
            "median": float(np.median(jerk)),
            "p90": float(np.percentile(jerk, 90)),
        },
        "gripper_close_edges": {
            "mean": float(np.mean(n_close)),
            "max": int(max(n_close)),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    stats = analyze(args.root)
    print(f"root: {stats['root']}")
    print(f"episodes: {stats['episodes']}  frames: {stats['frames']}")
    d = stats["duration_s"]
    print(f"duration_s: min={d['min']:.2f}  median={d['median']:.2f}  max={d['max']:.2f}")
    p = stats["ee_path_m"]
    print(f"ee_path_m:  median={p['median']:.3f}  p90={p['p90']:.3f}  max={p['max']:.3f}")
    j = stats["ee_jerk_proxy"]
    print(f"jerk_proxy: median={j['median']:.3f}  p90={j['p90']:.3f}")
    g = stats["gripper_close_edges"]
    print(f"gripper_close_edges: mean={g['mean']:.2f}  max={g['max']}")
    print(
        "note: cube x/y/yaw is not in this parquet (record used reset_cube=random). "
        "Cannot build a spawn coverage heatmap until auto_collect logs spawn."
    )


if __name__ == "__main__":
    main()
