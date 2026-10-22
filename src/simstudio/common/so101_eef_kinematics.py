"""SO-101 forward/inverse kinematics for Cartesian policy adapters.

LingBot-VA's LIBERO checkpoint predicts controller commands, not joints:
``[dx, dy, dz, d_axis_angle_x, d_axis_angle_y, d_axis_angle_z, gripper]``.
The translational and rotational channels are scaled by the LIBERO OSC
controller (0.05 m and 0.5 rad respectively).  This module keeps that model
contract outside the pinned LeRobot submodule and projects the requested
6-DoF pose increment onto SO-101's five arm joints.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

ARM_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)
GRIPPER_JOINT = "gripper"


@dataclass(frozen=True)
class IKSolveInfo:
    converged: bool
    iterations: int
    position_error_m: float
    orientation_error_rad: float


class SO101EEFAdapter:
    """Convert between SO-101 joints and LIBERO-style EEF commands."""

    def __init__(
        self,
        xml_path: str | Path = "SO101/scenes/simple_pick/scene.xml",
        ee_site_name: str = "wrist_site",
        *,
        translation_scale_m: float = 0.05,
        rotation_scale_rad: float = 0.5,
        gripper_open: float = 0.85,
        gripper_closed: float = -0.1,
    ) -> None:
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)
        self.translation_scale_m = float(translation_scale_m)
        self.rotation_scale_rad = float(rotation_scale_rad)
        self.gripper_open = float(gripper_open)
        self.gripper_closed = float(gripper_closed)
        self.gripper_midpoint = 0.5 * (self.gripper_open + self.gripper_closed)

        self.arm_joint_ids = np.asarray(
            [mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
            dtype=np.int32,
        )
        if np.any(self.arm_joint_ids < 0):
            raise RuntimeError(f"Missing SO-101 arm joints in {xml_path}")
        self.arm_qpos_ids = self.model.jnt_qposadr[self.arm_joint_ids].astype(np.int32)
        self.arm_dof_ids = self.model.jnt_dofadr[self.arm_joint_ids].astype(np.int32)
        self.arm_lower = self.model.jnt_range[self.arm_joint_ids, 0].copy()
        self.arm_upper = self.model.jnt_range[self.arm_joint_ids, 1].copy()
        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, ee_site_name)
        if self.ee_site_id < 0:
            raise RuntimeError(f"Missing end-effector site {ee_site_name!r} in {xml_path}")

    def forward(self, joints: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        """Return world-frame position and rotation matrix for five arm joints."""
        q = np.asarray(joints, dtype=np.float64)
        if q.shape[-1] < len(ARM_JOINTS):
            raise ValueError(f"Expected at least 5 joints, got shape {q.shape}")
        self.data.qpos[self.arm_qpos_ids] = q[: len(ARM_JOINTS)]
        mj.mj_forward(self.model, self.data)
        position = self.data.site_xpos[self.ee_site_id].copy()
        rotation = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        return position, rotation

    def joint_target_to_eef_command(
        self,
        current_joints: Sequence[float],
        target_joints: Sequence[float],
        *,
        clip: bool = True,
    ) -> np.ndarray:
        """Convert an absolute joint target into one LIBERO controller command."""
        current = np.asarray(current_joints, dtype=np.float64)
        target = np.asarray(target_joints, dtype=np.float64)
        if current.shape[-1] < 6 or target.shape[-1] < 6:
            raise ValueError("SO-101 state and action must each contain 6 joints")
        p0, r0 = self.forward(current)
        p1, r1 = self.forward(target)
        translation = (p1 - p0) / self.translation_scale_m
        # LIBERO left-multiplies a world-frame axis-angle increment.
        rotation = Rotation.from_matrix(r1 @ r0.T).as_rotvec() / self.rotation_scale_rad
        arm = np.concatenate((translation, rotation))
        if clip:
            arm = np.clip(arm, -1.0, 1.0)
        # LIBERO gripper commands are signs held every controller tick.
        gripper = 1.0 if target[5] <= self.gripper_midpoint else -1.0
        return np.asarray([*arm, gripper], dtype=np.float32)

    def eef_command_to_joint_target(
        self,
        current_joints: Sequence[float],
        command: Sequence[float],
        *,
        max_iterations: int = 80,
        damping: float = 1e-3,
        orientation_weight: float = 0.15,
        max_joint_step: float = 0.2,
    ) -> tuple[np.ndarray, IKSolveInfo]:
        """Project one LIBERO command onto SO-101's reachable joint manifold."""
        current = np.asarray(current_joints, dtype=np.float64)
        action = np.asarray(command, dtype=np.float64)
        if current.shape[-1] < 6:
            raise ValueError(f"Expected 6 current joints, got shape {current.shape}")
        if action.shape[-1] != 7:
            raise ValueError(f"Expected a 7-D EEF command, got shape {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("EEF command contains non-finite values")

        p0, r0 = self.forward(current)
        arm_action = np.clip(action[:6], -1.0, 1.0)
        target_position = p0 + arm_action[:3] * self.translation_scale_m
        target_rotation = (
            Rotation.from_rotvec(arm_action[3:6] * self.rotation_scale_rad).as_matrix() @ r0
        )
        q = np.clip(current[:5].copy(), self.arm_lower, self.arm_upper)
        position_error = float("inf")
        orientation_error = float("inf")
        converged = False
        iterations = 0

        for _ in range(max_iterations):
            iterations += 1
            position, rotation = self.forward(q)
            ep = target_position - position
            er = Rotation.from_matrix(target_rotation @ rotation.T).as_rotvec()
            position_error = float(np.linalg.norm(ep))
            orientation_error = float(np.linalg.norm(er))
            if position_error <= 2e-4 and orientation_error <= 3e-3:
                converged = True
                break

            jp = np.zeros((3, self.model.nv), dtype=np.float64)
            jr = np.zeros((3, self.model.nv), dtype=np.float64)
            mj.mj_jacSite(self.model, self.data, jp, jr, self.ee_site_id)
            jacobian = np.vstack(
                (
                    jp[:, self.arm_dof_ids],
                    orientation_weight * jr[:, self.arm_dof_ids],
                )
            )
            error = np.concatenate((ep, orientation_weight * er))
            normal = jacobian.T @ jacobian + (damping**2) * np.eye(len(ARM_JOINTS))
            dq = np.linalg.solve(normal, jacobian.T @ error)
            dq = np.clip(dq, -max_joint_step, max_joint_step)
            q = np.clip(q + dq, self.arm_lower, self.arm_upper)

        target = np.empty(6, dtype=np.float32)
        target[:5] = q
        target[5] = self.gripper_closed if action[6] >= 0.0 else self.gripper_open
        return target, IKSolveInfo(
            converged=converged,
            iterations=iterations,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
        )
