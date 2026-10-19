"""Bounded API for RAX-style SO-101 smart-drawer exploration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.robots.robot import Robot

import mujoco as mj
import numpy as np

from simstudio.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot


@dataclass(frozen=True)
class DrawerTaskConfig:
    scene_path: Path = Path("SO101/scenes/smart_drawer/scene.xml")
    camera_names: tuple[str, ...] = ("front", "top", "wrist")
    render_window: bool = False
    success_position_m: float = 0.06
    episode_limit_s: float = 45.0
    max_action_duration_s: float = 2.0
    max_linear_speed_mps: float = 0.04
    max_yaw_rate_rps: float = 1.2
    max_wrist_rate_rps: float = 6.0
    max_gripper_rate: float = 0.7


class RaxDrawerEnv:
    """Expose reset/observe/step/check_success around the existing SO-101 robot.

    Actions: vx/vy/vz in world-frame m/s; wrist_flex_rate/yaw_rate in rad/s;
    gripper_delta in the keyboard controller's rate convention.
    """

    ACTION_LIMITS = {
        "vx": "max_linear_speed_mps",
        "vy": "max_linear_speed_mps",
        "vz": "max_linear_speed_mps",
        "wrist_flex_rate": "max_wrist_rate_rps",
        "yaw_rate": "max_yaw_rate_rps",
        "gripper_delta": "max_gripper_rate",
    }

    def __init__(self, config: DrawerTaskConfig | None = None):
        assert issubclass(SO101MujocoRobot, Robot)
        self.task = config or DrawerTaskConfig()
        self.repo_root = Path(__file__).resolve().parents[2]
        self.scene_path = self.task.scene_path
        if not self.scene_path.is_absolute():
            self.scene_path = self.repo_root / self.scene_path
        if not self.scene_path.is_file():
            raise FileNotFoundError(self.scene_path)

        cfg = SO101MujocoConfig(
            xml_path=self.scene_path,
            cube_positions_path=self.repo_root / "configs/scenes/smart_drawer/cube_positions.json",
            camera_names=list(self.task.camera_names),
            render_window=self.task.render_window,
            action_mode="velocity",
            ee_site_name="gripperframe",
        )
        self.robot = SO101MujocoRobot(cfg)
        self.robot.connect(calibrate=False)
        jid = mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_JOINT, "drawer_slide")
        sid = mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_SENSOR, "drawer_position")
        camera_ids = {
            name: mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_CAMERA, name)
            for name in self.task.camera_names
        }
        if jid < 0 or sid < 0 or any(camera_id < 0 for camera_id in camera_ids.values()):
            self.robot.disconnect()
            raise ValueError("scene requires drawer_slide, drawer_position, and all configured cameras")
        self.qadr = int(self.robot.model.jnt_qposadr[jid])
        self.dadr = int(self.robot.model.jnt_dofadr[jid])
        self.sadr = int(self.robot.model.sensor_adr[sid])
        self.bounds = tuple(float(x) for x in self.robot.model.jnt_range[jid])
        if not 0 < self.task.success_position_m <= self.bounds[1] - self.bounds[0]:
            self.robot.disconnect()
            raise ValueError("success distance must be positive and within the drawer travel")
        positive_limits = (
            self.task.episode_limit_s,
            self.task.max_action_duration_s,
            self.task.max_linear_speed_mps,
            self.task.max_yaw_rate_rps,
            self.task.max_wrist_rate_rps,
            self.task.max_gripper_rate,
        )
        if not all(np.isfinite(value) and value > 0 for value in positive_limits):
            self.robot.disconnect()
            raise ValueError("episode, action, and control limits must be positive and finite")
        self.episode = -1
        self.start_time = 0.0
        self.seed: int | None = None
        self.start_drawer_position = 0.0
        self.closed = False
        self.terminated = False

    @property
    def model(self):
        return self.robot.model

    @property
    def data(self):
        return self.robot.data

    def _check_open(self):
        if self.closed or not self.robot.is_connected:
            raise RuntimeError("environment is closed")

    def drawer_position(self) -> float:
        self._check_open()
        return float(self.data.sensordata[self.sadr])

    def check_success(self) -> bool:
        opened = self.drawer_position() - self.start_drawer_position
        return opened + 1e-6 >= self.task.success_position_m

    def reset(
        self, *, seed: int | None = None, drawer_position_m: float = 0.0, include_cameras: bool = True
    ) -> dict[str, Any]:
        self._check_open()
        pos = float(drawer_position_m)
        max_start = self.bounds[1] - self.task.success_position_m
        if not np.isfinite(pos) or not self.bounds[0] <= pos <= max_start:
            raise ValueError(
                f"drawer_position_m must be in [{self.bounds[0]}, {max_start}] "
                "so the success distance remains reachable"
            )
        self.robot.reset_to_home_position()
        self.robot.reset_block_position(0)
        self.data.qpos[self.qadr] = pos
        self.data.qvel[self.dadr] = 0.0
        self.robot.q_des = self.data.qpos.copy()
        mj.mj_forward(self.model, self.data)
        for _ in range(max(1, round(0.08 / self.model.opt.timestep))):
            mj.mj_step(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        self.episode += 1
        self.seed = seed
        self.start_drawer_position = self.drawer_position()
        self.start_time = float(self.data.time)
        self.terminated = False
        return self.observe(include_cameras=include_cameras)

    def observe(self, *, include_cameras: bool = True) -> dict[str, Any]:
        self._check_open()
        mj.mj_forward(self.model, self.data)
        result: dict[str, Any] = {
            "episode": self.episode,
            "seed": self.seed,
            "sim_time_s": float(self.data.time - self.start_time),
            "drawer_position_m": self.drawer_position(),
            "drawer_opened_distance_m": self.drawer_position() - self.start_drawer_position,
            "drawer_range_m": list(self.bounds),
            "success_threshold_m": self.task.success_position_m,
            "success": self.check_success(),
            "joints": {
                n: {
                    "position_rad": float(self.data.qpos[self.robot.dof_ids[n]]),
                    "velocity_rad_s": float(self.data.qvel[self.robot.dof_ids[n]]),
                }
                for n in self.robot.JOINT_NAMES
            },
            "end_effector_position_m": [float(v) for v in self.data.site_xpos[self.robot.ee_site_id]],
        }
        if include_cameras:
            raw = self.robot.get_observation()
            result["cameras"] = {n: raw[f"camera_{n}"] for n in self.task.camera_names}
        return result

    def _bound_action(self, action: dict[str, float]) -> dict[str, float]:
        unknown = set(action) - set(self.ACTION_LIMITS)
        if unknown:
            raise ValueError(f"unknown action keys: {sorted(unknown)}")
        bounded = {}
        for key, value in action.items():
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError(f"action {key!r} must be finite")
            limit = getattr(self.task, self.ACTION_LIMITS[key])
            bounded[key] = float(np.clip(numeric, -limit, limit))
        return bounded

    def step(
        self, action: dict[str, float], *, duration_s: float = 0.25, include_cameras: bool = True
    ) -> dict[str, Any]:
        self._check_open()
        if self.episode < 0:
            raise RuntimeError("call reset() before step()")
        if self.terminated:
            raise RuntimeError("episode has terminated; call reset() before another step")
        duration = float(duration_s)
        if not np.isfinite(duration) or not 0 < duration <= self.task.max_action_duration_s:
            raise ValueError(f"duration_s must be in (0, {self.task.max_action_duration_s}]")
        bounded = self._bound_action(action)
        frames = max(1, round(duration * self.robot.config.record_fps))
        remaining = self.task.episode_limit_s - (self.data.time - self.start_time)
        if remaining <= 0:
            self.terminated = True
            obs = self.observe(include_cameras=include_cameras)
            return {
                "observation": obs,
                "action_applied": bounded,
                "terminated": True,
                "termination_reason": "time_limit",
                "success": False,
            }
        frames = min(frames, max(1, int(np.ceil(remaining * self.robot.config.record_fps))))
        for _ in range(frames):
            self.robot.send_action(bounded)
            if self.check_success() or self.data.time - self.start_time >= self.task.episode_limit_s:
                break
        obs = self.observe(include_cameras=include_cameras)
        reason = (
            "success"
            if obs["success"]
            else "time_limit"
            if obs["sim_time_s"] >= self.task.episode_limit_s
            else None
        )
        self.terminated = reason is not None
        return {
            "observation": obs,
            "action_applied": bounded,
            "terminated": self.terminated,
            "termination_reason": reason,
            "success": obs["success"],
        }

    def close(self):
        if not self.closed:
            self.robot.disconnect()
            self.closed = True

    def __enter__(self) -> RaxDrawerEnv:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
