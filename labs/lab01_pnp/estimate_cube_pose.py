#!/usr/bin/env python3
"""Recover per-episode cube reset pose from the top camera.

Only ``auto_collect_yaw`` logged cube pose to a sidecar. The teleop and
fixed-wrist agent campaigns randomized the same envelope (x 0.26-0.34,
y 0.165-0.235, yaw +-45 deg) but never persisted the draw, so their pose
coverage cannot be read from disk directly.

The cube is the only saturated green object in the scene, the ``top`` camera is
static, and its pose/FOV are known from the MJCF. So for each episode we take
the first frame, segment the cube, back-project the mask onto the cube's top
face plane, and fit a minimum-area rectangle: its center gives XY and its
orientation gives yaw (folded to +-45 deg, which is exactly the sampled range).

The mask also catches the two visible side faces, which biases the fitted
center by a fixed offset. That offset is calibrated against the agent-yaw
ground truth and then applied to every dataset. Residual accuracy on the
calibration set is reported by ``--report`` so the estimate is never presented
as if it were logged truth.

Usage::

    .venv-rocm/bin/python labs/lab01_pnp/estimate_cube_pose.py --report
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import av
import cv2
import mujoco as mj
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "datasets"
SCENE = REPO / "SO101" / "scenes" / "simple_pick" / "scene.xml"
OUT_JSON = REPO / "labs" / "lab01_pnp_agent" / "viz" / "cube_pose_estimates.json"

TELEOP = "so101-simstudio-lab01-pnp"
AGENT = "so101-simstudio-lab01-pnp-agent"
AGENT_YAW = "so101-simstudio-lab01-pnp-agent-yaw"

TOP_KEY = "observation.images.camera_top"
CUBE_TOP_Z = 0.030  # cube is 30 mm tall; its top face is the dominant surface
HSV_LO = (35, 90, 60)
HSV_HI = (85, 255, 255)
# Plausible fitted-rect extent: a 30 mm cube plus its visible side faces.
RECT_MM = (25.0, 70.0)
MIN_AREA_PX = 300


@dataclass
class CubePose:
    episode: int
    x: float
    y: float
    yaw_deg: float
    area_px: int
    rect_mm: tuple[float, float]
    ok: bool


class TopCamera:
    """Back-projects top-camera pixels onto a horizontal world plane."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        model = mj.MjModel.from_xml_path(str(SCENE))
        data = mj.MjData(model)
        mj.mj_forward(model, data)
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, "top")

        self.pos = data.cam_xpos[cam_id].copy()
        self.rot = data.cam_xmat[cam_id].reshape(3, 3).copy()
        self.width = width
        self.height = height
        self.focal = (height / 2) / np.tan(np.radians(model.cam_fovy[cam_id]) / 2)

    def to_plane(self, uv: np.ndarray, z: float) -> np.ndarray:
        u, v = uv[..., 0], uv[..., 1]
        cam_dir = np.stack(
            [
                (u - (self.width / 2 - 0.5)) / self.focal,
                -(v - (self.height / 2 - 0.5)) / self.focal,
                -np.ones_like(u),
            ],
            axis=-1,
        )
        world_dir = cam_dir @ self.rot.T
        scale = (z - self.pos[2]) / world_dir[..., 2]
        return self.pos[None, :] + scale[..., None] * world_dir


def first_frames(root: Path) -> list[np.ndarray]:
    """First RGB frame of every episode, read via per-episode video offsets."""
    meta_files = sorted((root / "meta" / "episodes").glob("**/*.parquet"))
    columns = [
        "episode_index",
        f"videos/{TOP_KEY}/chunk_index",
        f"videos/{TOP_KEY}/file_index",
        f"videos/{TOP_KEY}/from_timestamp",
    ]
    meta = [row for f in meta_files for row in pq.read_table(f, columns=columns).to_pylist()]
    meta.sort(key=lambda r: r["episode_index"])

    frames: list[np.ndarray] = []
    containers: dict[Path, av.container.InputContainer] = {}
    for row in meta:
        path = (
            root
            / "videos"
            / TOP_KEY
            / f"chunk-{row[f'videos/{TOP_KEY}/chunk_index']:03d}"
            / f"file-{row[f'videos/{TOP_KEY}/file_index']:03d}.mp4"
        )
        container = containers.get(path)
        if container is None:
            container = containers[path] = av.open(str(path))

        stream = container.streams.video[0]
        start = row[f"videos/{TOP_KEY}/from_timestamp"]
        container.seek(int(start / stream.time_base), stream=stream)
        frame = next(f for f in container.decode(stream) if f.time is not None and f.time >= start - 1e-3)
        frames.append(frame.to_ndarray(format="rgb24"))

    for container in containers.values():
        container.close()
    return frames


def estimate(frame: np.ndarray, cam: TopCamera, episode: int) -> CubePose:
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_LO, HSV_HI)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2:
        return CubePose(episode, float("nan"), float("nan"), float("nan"), 0, (0.0, 0.0), False)

    # The robot base picks up a green tint in some frames; keep the cube blob only.
    blob = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[blob, cv2.CC_STAT_AREA])
    ys, xs = np.nonzero(labels == blob)
    plane = cam.to_plane(np.stack([xs, ys], axis=-1).astype(float), CUBE_TOP_Z)[:, :2]

    (cx, cy), (w, h), angle = cv2.minAreaRect((plane * 1000).astype(np.float32))
    yaw = ((angle + 45.0) % 90.0) - 45.0
    ok = area >= MIN_AREA_PX and all(RECT_MM[0] <= s <= RECT_MM[1] for s in (w, h))
    return CubePose(episode, cx / 1000, cy / 1000, yaw, area, (round(w, 1), round(h, 1)), ok)


def load_truth() -> list[dict]:
    rows = (DATASETS / AGENT_YAW / "spawns.jsonl").read_text().splitlines()
    return [r for r in (json.loads(line) for line in rows if line.strip()) if r.get("success")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="store_true", help="print calibration residuals")
    args = parser.parse_args()

    cam = TopCamera()
    raw: dict[str, list[CubePose]] = {}
    for key, name in (("teleop", TELEOP), ("agent", AGENT), ("yaw", AGENT_YAW)):
        frames = first_frames(DATASETS / name)
        raw[key] = [estimate(f, cam, i) for i, f in enumerate(frames)]

    # Calibrate the side-face centroid bias on the only dataset with logged truth.
    truth = load_truth()
    if len(truth) != len(raw["yaw"]):
        raise RuntimeError(f"agent-yaw truth {len(truth)} != estimates {len(raw['yaw'])}")

    pairs = [(e, t) for e, t in zip(raw["yaw"], truth, strict=True) if e.ok]
    dx = np.array([e.x - t["spawn"][0] for e, t in pairs])
    dy = np.array([e.y - t["spawn"][1] for e, t in pairs])
    dyaw = np.array([((e.yaw_deg - t["yaw_deg"] + 45) % 90) - 45 for e, t in pairs])
    bias = (float(dx.mean()), float(dy.mean()))

    out = {
        "calibration": {
            "source": "agent-yaw spawns.jsonl ground truth",
            "n": len(pairs),
            "bias_xy_m": bias,
            "residual_x_mm_std": float((dx - dx.mean()).std() * 1000),
            "residual_y_mm_std": float((dy - dy.mean()).std() * 1000),
            "yaw_err_deg_mean": float(dyaw.mean()),
            "yaw_err_deg_std": float(dyaw.std()),
            "yaw_err_deg_max_abs": float(np.abs(dyaw).max()),
        },
        "datasets": {},
    }
    for key, poses in raw.items():
        out["datasets"][key] = [
            {**asdict(p), "x": p.x - bias[0], "y": p.y - bias[1]} if p.ok else asdict(p) for p in poses
        ]

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))

    if args.report:
        cal = out["calibration"]
        print(f"calibrated on {cal['n']} agent-yaw episodes with logged pose")
        print(f"  center bias removed : dx {bias[0] * 1000:+.2f} mm, dy {bias[1] * 1000:+.2f} mm")
        print(
            f"  residual after bias : x sigma {cal['residual_x_mm_std']:.2f} mm, "
            f"y sigma {cal['residual_y_mm_std']:.2f} mm"
        )
        print(
            f"  yaw error           : {cal['yaw_err_deg_mean']:+.2f} +- "
            f"{cal['yaw_err_deg_std']:.2f} deg, max |err| {cal['yaw_err_deg_max_abs']:.2f} deg"
        )
        for key, poses in raw.items():
            bad = [p.episode for p in poses if not p.ok]
            print(
                f"  {key:7s}: {len(poses) - len(bad)}/{len(poses)} episodes usable"
                + (f", rejected {bad}" if bad else "")
            )
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
