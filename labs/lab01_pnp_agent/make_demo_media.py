#!/usr/bin/env python3
"""Export demo clips and keyframe sheets from the agent-yaw LeRobot datasets.

Each clip is one recorded episode with the top, front, and wrist cameras side by
side (960x240, H.264). Each sheet is five evenly spaced frames per camera
(1600x720 JPEG). Outputs go to ``labs/lab01_pnp_agent/media/``.

    python3 labs/lab01_pnp_agent/make_demo_media.py
    python3 labs/lab01_pnp_agent/make_demo_media.py --only actfollow_ep171

Needs ``ffmpeg`` with an AV1 decoder (libdav1d) and libx264.
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

LAB = Path(__file__).resolve().parent
DATASETS = LAB / "datasets"
OUT = LAB / "media"
CAMERAS = ("camera_top", "camera_front", "camera_wrist")
VIEW_W, VIEW_H = 320, 240
SHEET_FRAMES = 5


@dataclass(frozen=True)
class Demo:
    name: str
    dataset: str
    episode: int
    caption: str


DEMOS = (
    Demo(
        "v4_scripted_ep000",
        "so101-simstudio-lab01-pnp-agent-yaw-v4",
        0,
        "agent-yaw v4 scripted expert, starts at home with gripper 0.85",
    ),
    Demo(
        "yaw_minus45_ep022",
        "so101-simstudio-lab01-pnp-agent-yaw",
        22,
        "agent-yaw, cube yaw -44.6 deg, wrist turns to the face",
    ),
    Demo(
        "yaw_plus42_ep026",
        "so101-simstudio-lab01-pnp-agent-yaw",
        26,
        "agent-yaw, cube yaw +42.3 deg (the band the fixed-wrist set lost)",
    ),
    Demo(
        "actfollow_ep171",
        "so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2",
        171,
        "v4 ACT 6-D rollout under reset_arm=follow, kept because it succeeded",
    ),
    Demo(
        "actfollow_ep192",
        "so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2",
        192,
        "last ACT-follow success added for SmolVLA follow2",
    ),
)


def _episode_row(root: Path, episode: int) -> dict:
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    table = pa.concat_tables([pq.read_table(f) for f in files])
    rows = table.to_pylist()
    for row in rows:
        if row["episode_index"] == episode:
            return row
    raise KeyError(f"episode {episode} not in {root}")


def _segment(root: Path, row: dict, camera: str) -> tuple[Path, float, float]:
    key = f"videos/observation.images.{camera}"
    path = (
        root
        / "videos"
        / f"observation.images.{camera}"
        / f"chunk-{row[f'{key}/chunk_index']:03d}"
        / f"file-{row[f'{key}/file_index']:03d}.mp4"
    )
    start = row[f"{key}/from_timestamp"]
    return path, start, row[f"{key}/to_timestamp"] - start


def _inputs(root: Path, row: dict) -> tuple[list[str], float]:
    args: list[str] = []
    duration = 0.0
    for camera in CAMERAS:
        path, start, length = _segment(root, row, camera)
        args += ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path)]
        duration = length
    return args, duration


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def export(demo: Demo) -> None:
    root = DATASETS / demo.dataset
    row = _episode_row(root, demo.episode)
    inputs, duration = _inputs(root, row)
    scale = ";".join(f"[{i}:v]scale={VIEW_W}:{VIEW_H}[v{i}]" for i in range(3))

    clip = OUT / f"{demo.name}.mp4"
    _run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            *inputs,
            "-filter_complex",
            f"{scale};[v0][v1][v2]hstack=inputs=3[out]",
            "-map",
            "[out]",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-preset",
            "slow",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(clip),
        ]
    )

    step = duration / SHEET_FRAMES
    with tempfile.TemporaryDirectory() as tmp:
        rows = []
        for i, camera in enumerate(CAMERAS):
            path, start, _ = _segment(root, row, camera)
            frames = []
            for k in range(SHEET_FRAMES):
                t = start + min(k * step + step / 2, duration - 0.1)
                if k == 0:
                    t = start
                frame = Path(tmp) / f"{i}_{k}.png"
                _run(
                    [
                        "ffmpeg",
                        "-y",
                        "-loglevel",
                        "error",
                        "-ss",
                        f"{t:.3f}",
                        "-i",
                        str(path),
                        "-frames:v",
                        "1",
                        "-vf",
                        f"scale={VIEW_W}:{VIEW_H}",
                        str(frame),
                    ]
                )
                frames.append(frame)
            strip = Path(tmp) / f"row{i}.png"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    *[a for f in frames for a in ("-i", str(f))],
                    "-filter_complex",
                    f"hstack=inputs={SHEET_FRAMES}",
                    str(strip),
                ]
            )
            rows.append(strip)
        _run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                *[a for r in rows for a in ("-i", str(r))],
                "-filter_complex",
                "vstack=inputs=3",
                "-q:v",
                "3",
                str(OUT / f"{demo.name}_frames.jpg"),
            ]
        )
    print(f"{demo.name}: {row['length']} frames, {duration:.1f} s -> {clip.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", nargs="*", help="demo names to export")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    for demo in DEMOS:
        if args.only and demo.name not in args.only:
            continue
        export(demo)


if __name__ == "__main__":
    main()
