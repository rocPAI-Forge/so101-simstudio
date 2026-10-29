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
    camera_names: tuple[str, ...] = ("top", "side", "wrist")
    render_window: bool = False
    success_position_m: float = 0.06
    close_tolerance_m: float = 0.005
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
            camera_resolutions={"top": (1280, 800)},
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
        self.camera_ids = camera_ids
        self.qadr = int(self.robot.model.jnt_qposadr[jid])
        self.dadr = int(self.robot.model.jnt_dofadr[jid])
        self.sadr = int(self.robot.model.sensor_adr[sid])
        self._depth_renderer = None
        if "top" in camera_ids:
            height, width = self.robot._camera_resolution("top")
            self._depth_renderer = mj.Renderer(self.robot.model, height=height, width=width)
            self._depth_renderer.enable_depth_rendering()
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
        rig = mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_BODY, "drawer_rig")
        handle = mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_GEOM, "drawer_handle_geom")
        gripper_bodies = {
            mj.mj_name2id(self.robot.model, mj.mjtObj.mjOBJ_BODY, name)
            for name in ("gripper", "moving_jaw_so101_v1")
        }
        if rig < 0 or handle < 0 or any(body < 0 for body in gripper_bodies):
            self.robot.disconnect()
            raise ValueError("scene requires drawer_rig, drawer_handle_geom, and gripper bodies")
        self.rig_mocap = int(self.robot.model.body_mocapid[rig])
        if self.rig_mocap < 0:
            self.robot.disconnect()
            raise ValueError("drawer_rig must be a mocap body")
        self.handle_geom = int(handle)
        self.gripper_geoms = {
            index
            for index in range(self.robot.model.ngeom)
            if int(self.robot.model.geom_bodyid[index]) in gripper_bodies
        }
        self.layout = {"x_m": 0.0, "y_m": 0.0, "yaw_deg": 0.0}
        self._reset_score()
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

    def set_layout(self, *, x_m: float = 0.0, y_m: float = 0.0, yaw_deg: float = 0.0) -> None:
        """Place the cabinet rig. The controller does not receive this pose."""
        self._check_open()
        values = {"x_m": float(x_m), "y_m": float(y_m), "yaw_deg": float(yaw_deg)}
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError("layout values must be finite")
        if abs(values["x_m"]) > 0.03 or abs(values["y_m"]) > 0.03 or abs(values["yaw_deg"]) > 10.0:
            raise ValueError("layout is outside the locked randomization range")
        self.layout = values

    def _apply_layout(self) -> None:
        yaw = np.deg2rad(self.layout["yaw_deg"])
        half = 0.5 * yaw
        self.data.mocap_pos[self.rig_mocap] = (self.layout["x_m"], self.layout["y_m"], 0.0)
        self.data.mocap_quat[self.rig_mocap] = (np.cos(half), 0.0, 0.0, np.sin(half))

    def _reset_score(self) -> None:
        self.reached_open = False
        self.pull_contact = False
        self.close_contact = False

    def _gripper_touching_handle(self) -> bool:
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if self.handle_geom in pair and pair & self.gripper_geoms:
                return True
        return False

    def _update_score(self) -> None:
        opened = self.drawer_position() - self.start_drawer_position
        touching = self._gripper_touching_handle()
        if opened + 1e-6 >= self.task.success_position_m:
            self.reached_open = True
        if touching and opened > 0.01:
            self.pull_contact = True
        if touching and self.reached_open and opened < self.task.success_position_m:
            self.close_contact = True

    def check_success(self) -> bool:
        if self.episode < 0:
            return False
        opened = abs(self.drawer_position() - self.start_drawer_position)
        return (
            self.reached_open
            and self.pull_contact
            and self.close_contact
            and opened <= self.task.close_tolerance_m + 1e-6
        )

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
        self._apply_layout()
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
        self._reset_score()
        return self.observe(include_cameras=include_cameras)

    def camera_calibration(self) -> dict[str, Any]:
        """Intrinsics and camera poses. These describe the sensors, not the drawer."""
        self._check_open()
        calibrated = {}
        for name, camera_id in self.camera_ids.items():
            height, width = self.robot._camera_resolution(name)
            fovy = float(self.model.cam_fovy[camera_id])
            focal = (height / 2.0) / np.tan(np.deg2rad(fovy) / 2.0)
            rotation = np.array(self.data.cam_xmat[camera_id], dtype=float).reshape(3, 3).T
            calibrated[name] = {
                "width": int(width),
                "height": int(height),
                "fovy_deg": fovy,
                "fx": float(focal),
                "fy": float(focal),
                "cx": float(width) / 2.0,
                "cy": float(height) / 2.0,
                "position_m": [float(v) for v in self.data.cam_xpos[camera_id]],
                "rotation_camera_to_world": [float(v) for v in rotation.reshape(-1)],
            }
        return calibrated

    def top_depth_m(self) -> np.ndarray:
        """Metric depth from the overhead camera, in meters along the camera ray."""
        self._check_open()
        if self._depth_renderer is None:
            raise RuntimeError("top depth camera is not configured")
        self._depth_renderer.update_scene(self.data, camera="top")
        return np.asarray(self._depth_renderer.render(), dtype=np.float32).copy()

    def observe(self, *, include_cameras: bool = True) -> dict[str, Any]:
        self._check_open()
        mj.mj_forward(self.model, self.data)
        result: dict[str, Any] = {
            "episode": self.episode,
            "seed": self.seed,
            "sim_time_s": float(self.data.time - self.start_time),
            "joints": {
                n: {
                    "position_rad": float(self.data.qpos[self.robot.dof_ids[n]]),
                    "velocity_rad_s": float(self.data.qvel[self.robot.dof_ids[n]]),
                }
                for n in self.robot.JOINT_NAMES
            },
            "calibration": self.camera_calibration(),
        }
        if include_cameras:
            raw = self.robot.get_observation()
            result["cameras"] = {n: raw[f"camera_{n}"] for n in self.task.camera_names}
            if self._depth_renderer is not None:
                depth_mm = np.clip(np.rint(self.top_depth_m() * 1000.0), 0, 65535).astype(np.uint16)
                result["depth_images"] = {"top": depth_mm}
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
            self._update_score()
            if self.check_success() or self.data.time - self.start_time >= self.task.episode_limit_s:
                break
        obs = self.observe(include_cameras=include_cameras)
        succeeded = self.check_success()
        reason = (
            "success"
            if succeeded
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
            "success": succeeded,
        }

    def close(self):
        if not self.closed:
            if self._depth_renderer is not None:
                self._depth_renderer.close()
                self._depth_renderer = None
            self.robot.disconnect()
            self.closed = True

    def __enter__(self) -> RaxDrawerEnv:
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
