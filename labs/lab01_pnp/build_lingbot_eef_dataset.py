#!/usr/bin/env python3
"""Build a local LingBot-VA EEF-label view of a Lab 01 joint dataset.

Videos and unchanged metadata are hard-linked. Parquet actions and their
statistics are rewritten, leaving the source dataset untouched. The output is
scratch data and must be validated before publishing.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from simstudio.common.so101_eef_kinematics import SO101EEFAdapter

ACTION = "action"
STATE = "observation.state"
STAT_KEYS = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")


def _stats(values: np.ndarray) -> dict[str, list[float] | list[int]]:
    quantiles = np.quantile(values, [0.01, 0.10, 0.50, 0.90, 0.99], axis=0)
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": quantiles[0].tolist(),
        "q10": quantiles[1].tolist(),
        "q50": quantiles[2].tolist(),
        "q90": quantiles[3].tolist(),
        "q99": quantiles[4].tolist(),
    }


def _atomic_write_table(table: pa.Table, path: Path) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    pq.write_table(table, tmp)
    os.replace(tmp, path)


def _atomic_write_json(payload: dict, path: Path) -> None:
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _replace_column(table: pa.Table, name: str, values: list[list[float]]) -> pa.Table:
    index = table.column_names.index(name)
    column_type = table.schema.field(name).type
    if pa.types.is_list(column_type) or pa.types.is_fixed_size_list(column_type):
        column_type = pa.list_(column_type.value_type)
    array = pa.array(values, type=column_type)
    return table.set_column(index, name, array)


def build_dataset(source: Path, output: Path, xml_path: Path, *, overwrite: bool = False) -> None:
    if source == output or source.is_relative_to(output) or output.is_relative_to(source):
        raise ValueError("Source and output dataset roots must be separate sibling trees")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)
    if not (source / "meta/info.json").is_file():
        raise FileNotFoundError(f"Not a LeRobot dataset root: {source}")

    shutil.copytree(source, output, copy_function=os.link)
    adapter = SO101EEFAdapter(xml_path)
    all_actions: list[np.ndarray] = []
    all_episode_ids: list[np.ndarray] = []

    for parquet_path in sorted((output / "data").glob("**/*.parquet")):
        table = pq.read_table(parquet_path)
        states = np.asarray(table[STATE].to_pylist(), dtype=np.float64)
        joint_actions = np.asarray(table[ACTION].to_pylist(), dtype=np.float64)
        if states.shape[1] < 6 or joint_actions.shape[1] != 6:
            raise ValueError(
                f"{parquet_path}: expected state>=6/action=6, got {states.shape}/{joint_actions.shape}"
            )
        eef_actions = np.stack(
            [
                adapter.joint_target_to_eef_command(state[:6], target, clip=True)
                for state, target in zip(states, joint_actions, strict=True)
            ]
        )
        table = _replace_column(table, STATE, states.astype(np.float32).tolist())
        table = _replace_column(table, ACTION, eef_actions.tolist())
        _atomic_write_table(table, parquet_path)
        all_actions.append(eef_actions)
        all_episode_ids.append(np.asarray(table["episode_index"].to_pylist(), dtype=np.int64))
        print(f"rewrote {parquet_path.relative_to(output)}: {len(eef_actions)} frames")

    actions = np.concatenate(all_actions)
    episode_ids = np.concatenate(all_episode_ids)
    global_stats = _stats(actions)

    info_path = output / "meta/info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["features"][ACTION] = {
        "dtype": "float32",
        "shape": [7],
        "names": [
            "eef.delta_x",
            "eef.delta_y",
            "eef.delta_z",
            "eef.delta_axis_angle_x",
            "eef.delta_axis_angle_y",
            "eef.delta_axis_angle_z",
            "gripper.command",
        ],
    }
    _atomic_write_json(info, info_path)

    stats_path = output / "meta/stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats[ACTION] = global_stats
    _atomic_write_json(stats, stats_path)

    for episodes_path in sorted((output / "meta/episodes").glob("**/*.parquet")):
        table = pq.read_table(episodes_path)
        rows = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
        for key in STAT_KEYS:
            name = f"stats/action/{key}"
            values = []
            for episode_id in rows:
                episode_values = actions[episode_ids == episode_id]
                values.append(_stats(episode_values)[key])
            table = _replace_column(table, name, values)
        _atomic_write_table(table, episodes_path)

    clipped = np.mean(np.abs(actions[:, :6]) >= 1.0, axis=0)
    print(f"built {output}: {len(actions)} frames")
    print("action q01:", np.round(np.quantile(actions, 0.01, axis=0), 5).tolist())
    print("action q99:", np.round(np.quantile(actions, 0.99, axis=0), 5).tolist())
    print("arm channels clipped fraction:", np.round(clipped, 6).tolist())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--xml",
        type=Path,
        default=Path("SO101/scenes/simple_pick/scene.xml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    build_dataset(
        args.source.resolve(),
        args.output.resolve(),
        args.xml.resolve(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
