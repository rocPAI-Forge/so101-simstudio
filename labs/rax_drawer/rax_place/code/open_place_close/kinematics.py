"""Forward kinematics from joint observations only."""

from __future__ import annotations

from pathlib import Path

import mujoco as mj
import numpy as np

_XML = Path(__file__).resolve().parents[5] / "SO101" / "scenes" / "smart_drawer" / "scene.xml"
_MODEL: mj.MjModel | None = None
_DATA: mj.MjData | None = None
_ARM = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _ensure() -> tuple[mj.MjModel, mj.MjData]:
    global _MODEL, _DATA
    if _MODEL is None:
        _MODEL = mj.MjModel.from_xml_path(str(_XML))
        _DATA = mj.MjData(_MODEL)
    return _MODEL, _DATA


def end_effector(observation: dict) -> np.ndarray:
    """Return gripperframe position from observed joint positions."""
    model, data = _ensure()
    joints = observation["joints"]
    for name in _ARM:
        jid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[jid]] = float(joints[name]["position_rad"])
    mj.mj_forward(model, data)
    sid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, "gripperframe")
    return data.site_xpos[sid].copy()
