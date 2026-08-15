#!/usr/bin/env python3
"""Render SO-101 eval cameras with a given MUJOCO_GL backend (osmesa/egl/glfw)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np


def main() -> None:
    if len(sys.argv) < 3:
        print("Usage: render_headless_cameras.py <backend> <output_dir>", file=sys.stderr)
        sys.exit(2)

    backend = sys.argv[1]
    out_dir = Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    os.environ["MUJOCO_GL"] = backend

    import mujoco  # noqa: E402

    model = mujoco.MjModel.from_xml_path("SO101/scenes/simple_pick/scene.xml")
    data = mujoco.MjData(model)

    # Home arm qpos (matches robot init)
    home = {
        "shoulder_pan": 0.0,
        "shoulder_lift": -0.3,
        "elbow_flex": 0.6,
        "wrist_flex": 1.2,
        "wrist_roll": 0.0,
        "gripper": -0.1,
    }
    for jname, val in home.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        data.qpos[model.jnt_qposadr[jid]] = val

    # Random cube similar to eval episode 3
    x, y, z, yaw_deg = 0.294, 0.182, 0.0125, 5.0
    yaw = np.deg2rad(yaw_deg)
    block_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
    adr = model.jnt_qposadr[block_jid]
    data.qpos[adr : adr + 3] = [x, y, z]
    data.qpos[adr + 3 : adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    for cam in ("front", "top", "wrist"):
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        path = out_dir / f"{backend}_{cam}.png"
        iio.imwrite(path, img)
        print(f"{backend} {cam}: {path} mean={float(img.mean()):.1f}")


if __name__ == "__main__":
    main()
