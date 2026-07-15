"""SO-101 MuJoCo robot implementation.

Adapted from robopicker/lerobot/src/lerobot/robots/so101_mujoco/robot_so101_mujoco.py
for latest HuggingFace LeRobot.
"""

import json
import logging
import warnings
from functools import cached_property
from pathlib import Path
from typing import Any

import glfw
import mujoco as mj
import mujoco.viewer
import numpy as np
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError

from simstudio.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig

logger = logging.getLogger(__name__)


class SO101MujocoRobot(Robot):
    """SO-101 robot in MuJoCo simulation."""

    config_class = SO101MujocoConfig
    name = "so101_mujoco"

    # Joint names (order matters)
    JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def __init__(self, config: SO101MujocoConfig):
        super().__init__(config)
        self.config = config

        # MuJoCo model and data (initially None until connect())
        self.model: mj.MjModel | None = None
        self.data: mj.MjData | None = None
        self._renderers: dict[str, mj.Renderer] = {}
        self._viewer = None

        # GLFW rendering
        self._glfw_window = None
        self._glfw_cam = None
        self._glfw_opt = None
        self._glfw_scene = None
        self._glfw_ctx = None
        self._glfw_initialized = False

        # Joint/actuator/site IDs (set in connect())
        self.dof_ids: dict[str, int] = {}
        self.act_ids: dict[str, int] = {}
        self.ee_site_id: int = -1
        self.robot_qpos_indices: np.ndarray | None = None

        # Control state
        self.q_des: np.ndarray | None = None
        self.dq_filt: np.ndarray | None = None
        self.j_lo: np.ndarray | None = None
        self.j_hi: np.ndarray | None = None
        self._mujoco_joint_range: dict[str, tuple[float, float]] = {}

        # Timing
        self.control_dt = 1.0 / config.control_fps
        self.physics_dt = 1.0 / config.physics_fps
        self.n_physics_per_control = int(self.control_dt / self.physics_dt)
        self.n_control_per_record = int((1.0 / config.record_fps) / self.control_dt)

        # Camera tracking for lerobot_record
        self.cameras = {f"camera_{name}": None for name in config.camera_names}

        # Load cube positions
        self.cube_positions = self._load_cube_positions()

        logger.info(
            "SO101MujocoRobot initialized: "
            f"record={config.record_fps}Hz, control={config.control_fps}Hz, "
            f"physics={config.physics_fps}Hz"
        )

    def _load_cube_positions(self) -> list[dict]:
        """Load cube positions from JSON configuration file."""
        cube_positions_path = self.config.cube_positions_path

        if cube_positions_path is None:
            logger.warning("No cube_positions_path specified - block positioning will not be available")
            return []

        if not cube_positions_path.exists():
            logger.warning(f"Cube positions file not found at {cube_positions_path}, using empty list")
            return []

        try:
            with open(cube_positions_path) as f:
                data = json.load(f)
                positions = data.get("cube_positions", [])
                logger.info(f"Loaded {len(positions)} cube positions from {cube_positions_path}")
                return positions
        except Exception as e:
            logger.error(f"Error loading cube positions from {cube_positions_path}: {e}")
            return []

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        """Define observation structure for dataset creation."""
        features: dict[str, type | tuple] = {
            "shoulder_pan.pos": float,
            "shoulder_lift.pos": float,
            "elbow_flex.pos": float,
            "wrist_flex.pos": float,
            "wrist_roll.pos": float,
            "gripper.pos": float,
            "shoulder_pan.vel": float,
            "shoulder_lift.vel": float,
            "elbow_flex.vel": float,
            "wrist_flex.vel": float,
            "wrist_roll.vel": float,
            "gripper.vel": float,
            "ee.pos_x": float,
            "ee.pos_y": float,
            "ee.pos_z": float,
        }
        for cam_name in self.config.camera_names:
            features[f"camera_{cam_name}"] = (self.config.camera_height, self.config.camera_width, 3)
        return features

    @cached_property
    def action_features(self) -> dict[str, type]:
        """Define action structure based on action_mode config."""
        if self.config.action_mode == "position":
            return {f"{joint}.pos": float for joint in self.JOINT_NAMES}
        else:
            return {
                "vx": float,
                "vy": float,
                "vz": float,
                "wrist_flex_rate": float,
                "yaw_rate": float,
                "gripper_delta": float,
            }

    @property
    def is_connected(self) -> bool:
        """Check if robot is connected (model loaded)."""
        return self.model is not None and self.data is not None

    def connect(self, calibrate: bool = True) -> None:
        """Load MuJoCo model and initialize control state."""
        if self.is_connected:
            logger.warning(f"{self} already connected")
            return

        xml_path = Path(self.config.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")

        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

        # Override physics timestep
        self.model.opt.timestep = self.physics_dt

        # Setup renderers for each camera
        for cam_name in self.config.camera_names:
            self._renderers[cam_name] = mj.Renderer(
                self.model,
                height=self.config.camera_height,
                width=self.config.camera_width,
            )

        # Map joint/actuator/site IDs
        self._setup_ids()

        # Get joint limits
        self.j_lo = self.model.jnt_range[:, 0].copy()
        self.j_hi = self.model.jnt_range[:, 1].copy()

        # Initialize control state
        self.dq_filt = np.zeros(self.model.nv)

        # Find and set home position
        self._initialize_home_position()

        # Initialize GLFW rendering
        self._init_glfw_rendering()

        logger.info(f"{self} connected successfully")

    def launch_viewer(self) -> None:
        """Launch the MuJoCo GLFW viewer window."""
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot must be connected before launching viewer")

        if self._viewer is not None:
            logger.warning("Viewer already launched")
            return

        self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._viewer.sync()
        logger.info("MuJoCo viewer window opened")

    def _setup_ids(self):
        """Map joint, actuator, and site names to MuJoCo IDs."""
        for joint_name in self.JOINT_NAMES:
            self.dof_ids[joint_name] = self.model.jnt_dofadr[
                mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
            ]
            self.act_ids[joint_name] = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, joint_name)
            # Extract ctrlrange from actuator for position scaling
            act_id = self.act_ids[joint_name]
            self._mujoco_joint_range[joint_name] = (
                float(self.model.actuator_ctrlrange[act_id, 0]),
                float(self.model.actuator_ctrlrange[act_id, 1]),
            )

        self.robot_qpos_indices = np.array([self.dof_ids[name] for name in self.JOINT_NAMES])

        # Joint id (not dof index) for the base pivot, used by the cylindrical
        # velocity transform to read the true shoulder_pan world anchor.
        self._shoulder_pan_jid = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_JOINT, "shoulder_pan"
        )

        self.ee_site_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SITE, self.config.ee_site_name)
        if self.ee_site_id < 0:
            raise RuntimeError(f"Site '{self.config.ee_site_name}' not found in model")

    def _initialize_home_position(self):
        """Set a good home position with tool pointing down."""
        q_home = self.data.qpos.copy()
        q_home[self.dof_ids["shoulder_pan"]] = 0.0
        q_home[self.dof_ids["shoulder_lift"]] = -0.3
        q_home[self.dof_ids["elbow_flex"]] = 0.6
        q_home[self.dof_ids["wrist_flex"]] = 1.2
        q_home[self.dof_ids["wrist_roll"]] = 0.0
        q_home[self.dof_ids["gripper"]] = self.config.home_gripper

        self.data.qpos[:] = q_home
        self.data.qvel[:] = 0.0
        self.q_des = q_home.copy()

        for joint_name in self.JOINT_NAMES:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        mj.mj_forward(self.model, self.data)

        robot_q = [q_home[self.dof_ids[name]] for name in self.JOINT_NAMES]
        ee_pos = self.data.site_xpos[self.ee_site_id]
        logger.info(f"Home position initialized: {robot_q}")
        logger.info(f"Home EE position: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]")

    def _init_glfw_rendering(self):
        """Initialize GLFW window and rendering context."""
        if not self.config.render_window:
            logger.info("GLFW rendering disabled by config")
            return

        if not glfw.init():
            logger.warning("GLFW init failed - running without visualization")
            return

        self._glfw_initialized = True
        window_width, window_height = 1280, 720

        # Ubuntu 24.04 / GNOME fix: use default window hints, do NOT set
        # FOCUSED=FALSE or FOCUS_ON_SHOW=FALSE or the window may be invisible.
        glfw.default_window_hints()
        self._glfw_window = glfw.create_window(
            window_width,
            window_height,
            "SO-101 MuJoCo Recording",
            None,
            None,
        )
        if not self._glfw_window:
            glfw.terminate()
            self._glfw_initialized = False
            logger.warning("Failed to create GLFW window - running without visualization")
            return

        self._glfw_cam = mj.MjvCamera()
        self._glfw_opt = mj.MjvOption()
        mj.mjv_defaultCamera(self._glfw_cam)
        self._glfw_cam.distance = 1.3
        self._glfw_cam.azimuth = 140
        self._glfw_cam.elevation = -20

        self._glfw_scene = mj.MjvScene(self.model, maxgeom=10000)

        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(1)
        self._glfw_ctx = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150)
        glfw.make_context_current(None)

        logger.info("GLFW visualization window created")

    def _render_glfw(self):
        """Render camera views to GLFW window in a grid layout."""
        if self._glfw_window is None:
            return

        if glfw.window_should_close(self._glfw_window):
            return

        glfw.make_context_current(self._glfw_window)
        glfw.swap_interval(1)

        viewport_width, viewport_height = glfw.get_framebuffer_size(self._glfw_window)
        cam_width = viewport_width // 2
        cam_height = viewport_height // 2

        camera_positions = {
            "top": (0, cam_height),
            "front": (cam_width, cam_height),
            "wrist": (0, 0),
        }

        for cam_name in self.config.camera_names:
            if cam_name not in camera_positions:
                continue

            x, y = camera_positions[cam_name]
            viewport = mj.MjrRect(x, y, cam_width, cam_height)

            mj.mjv_updateScene(
                self.model,
                self.data,
                self._glfw_opt,
                None,
                self._glfw_cam,
                mj.mjtCatBit.mjCAT_ALL,
                self._glfw_scene,
            )

            cam_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_CAMERA, cam_name)
            if cam_id >= 0:
                self._glfw_cam.type = mj.mjtCamera.mjCAMERA_FIXED
                self._glfw_cam.fixedcamid = cam_id
                mj.mjv_updateScene(
                    self.model,
                    self.data,
                    self._glfw_opt,
                    None,
                    self._glfw_cam,
                    mj.mjtCatBit.mjCAT_ALL,
                    self._glfw_scene,
                )

            mj.mjr_render(viewport, self._glfw_scene, self._glfw_ctx)

        glfw.swap_buffers(self._glfw_window)
        glfw.poll_events()
        glfw.make_context_current(None)

    @property
    def is_calibrated(self) -> bool:
        """Simulation doesn't need calibration."""
        return True

    def calibrate(self) -> None:
        """No-op for simulation."""
        pass

    def configure(self) -> None:
        """No-op for simulation."""
        pass

    def get_observation(self) -> RobotObservation:
        """Get current robot state."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        mj.mj_forward(self.model, self.data)

        obs: dict[str, Any] = {}
        for joint_name in self.JOINT_NAMES:
            dof_id = self.dof_ids[joint_name]
            obs[f"{joint_name}.pos"] = float(self.data.qpos[dof_id])
            obs[f"{joint_name}.vel"] = float(self.data.qvel[dof_id])

        ee_pos = self.data.site_xpos[self.ee_site_id]
        obs["ee.pos_x"] = float(ee_pos[0])
        obs["ee.pos_y"] = float(ee_pos[1])
        obs["ee.pos_z"] = float(ee_pos[2])

        for cam_name in self.config.camera_names:
            obs[f"camera_{cam_name}"] = self._render_camera(cam_name)

        return obs

    def _render_camera(self, camera_name: str) -> np.ndarray:
        """Render camera view."""
        if camera_name not in self._renderers:
            return np.zeros(
                (self.config.camera_height, self.config.camera_width, 3),
                dtype=np.uint8,
            )

        renderer = self._renderers[camera_name]
        renderer.update_scene(self.data, camera=camera_name)
        pixels = renderer.render()
        return pixels.astype(np.uint8)

    def send_action(self, action: RobotAction) -> RobotAction:
        """Main action dispatch - auto-detects action type and routes appropriately."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        has_velocity_keys = any(k in action for k in ["vx", "vy", "vz", "yaw_rate", "gripper_delta"])
        has_position_keys = any(f"{jn}.pos" in action for jn in self.JOINT_NAMES)

        if has_position_keys:
            return self._send_action_position(action)
        elif has_velocity_keys or not action:
            return self._send_action_teleop(action)
        else:
            raise ValueError(
                f"Unknown action format. Expected either velocity keys (vx, vy, vz) "
                f"or position keys ({self.JOINT_NAMES[0]}.pos, ...). Got: {list(action.keys())}"
            )

    def _send_action_teleop(self, action: RobotAction) -> RobotAction:
        """Velocity-based control for teleoperation."""
        if action:
            vx = action.get("vx", 0.0)
            vy = action.get("vy", 0.0)
            vz = action.get("vz", 0.0)
            yaw_rate = action.get("yaw_rate", 0.0)
            wrist_flex_rate = action.get("wrist_flex_rate", 0.0)
            gripper_delta = action.get("gripper_delta", 0.0)
        else:
            vx = vy = vz = yaw_rate = wrist_flex_rate = gripper_delta = 0.0

        for _ in range(self.n_control_per_record):
            self._control_step(vx, vy, vz, yaw_rate, wrist_flex_rate, gripper_delta)

        self._render_glfw()

        action_to_record = {}
        for joint_name in self.JOINT_NAMES:
            action_to_record[f"{joint_name}.pos"] = float(self.q_des[self.dof_ids[joint_name]])
        return action_to_record

    def _send_action_position(self, action: RobotAction) -> RobotAction:
        """Position-based control for replay/policy execution.

        Supports two input ranges:
        - Radians (from replay/policy): applied directly
        - Normalized (from leader arm -100..+100 / 0..100): scaled to MuJoCo ctrlrange

        Returns the action actually sent (scaled to radians) so the dataset
        records consistent values.
        """
        scaled_action = {}
        for joint_name in self.JOINT_NAMES:
            key = f"{joint_name}.pos"
            if key not in action:
                continue
            val = float(action[key])
            # Detect if value is in normalized range (abs > pi suggests normalized)
            if abs(val) > 4.0:
                # Normalized leader arm range: scale to MuJoCo ctrlrange
                if joint_name == "gripper":
                    lo, hi = 0.0, 100.0
                else:
                    lo, hi = -100.0, 100.0
                mj_lo, mj_hi = self._mujoco_joint_range[joint_name]
                t = (val - lo) / (hi - lo)
                val = mj_lo + t * (mj_hi - mj_lo)
            self.data.ctrl[self.act_ids[joint_name]] = val
            scaled_action[key] = val

        n_physics_steps = int((1.0 / self.config.record_fps) / self.physics_dt)
        for _ in range(n_physics_steps):
            mj.mj_step(self.model, self.data)

        self._render_glfw()
        return scaled_action

    def _control_step(
        self,
        vx: float,
        vy: float,
        vz: float,
        yaw_rate: float,
        wrist_flex_rate: float,
        gripper_delta: float,
    ):
        """Single control iteration with manual wrist control."""
        mj.mj_forward(self.model, self.data)

        Jp = np.zeros((3, self.model.nv))
        Jr = np.zeros((3, self.model.nv))
        mj.mj_jacSite(self.model, self.data, Jp, Jr, self.ee_site_id)

        arm_cols = [
            self.dof_ids["shoulder_pan"],
            self.dof_ids["shoulder_lift"],
            self.dof_ids["elbow_flex"],
        ]
        J3 = Jp[:, arm_cols]
        vx_w, vy_w = vx, vy
        if self.config.horizontal_control_mode == "cylindrical":
            # vx = radial reach (outward+), vy = tangential base swing.
            # Radial must be measured from the TRUE base pivot (shoulder_pan world
            # anchor); otherwise the commanded reach direction leaves the arm's
            # reach plane and the IK has to rotate the base to follow it, so pure
            # "forward" would inconsistently extend or swing.
            base = self.data.xanchor[self._shoulder_pan_jid]
            base_x = self.config.base_xy[0] or base[0]
            base_y = self.config.base_xy[1] or base[1]
            ee = self.data.site_xpos[self.ee_site_id]
            rx = ee[0] - base_x
            ry = ee[1] - base_y
            r = float(np.hypot(rx, ry))
            if r > 1e-6:
                ux, uy = rx / r, ry / r  # radial (outward) unit
            else:
                ux, uy = 1.0, 0.0
            tx, ty = -uy, ux  # tangential (CCW) unit
            vx_w = ux * vx + tx * vy
            vy_w = uy * vx + ty * vy
        v_des = np.array([vx_w, vy_w, vz])
        A = J3 @ J3.T + (self.config.lambda_pos**2) * np.eye(3)
        dq3 = J3.T @ np.linalg.solve(A, v_des)
        dq = np.zeros(self.model.nv)
        dq[arm_cols] = dq3

        dq[self.dof_ids["wrist_flex"]] = wrist_flex_rate
        dq[self.dof_ids["wrist_roll"]] += yaw_rate

        dq_lim = self.config.vel_limit * np.ones(self.model.nv)
        dq_lim[self.dof_ids["wrist_flex"]] = self.config.vel_limit_wrist
        dq_lim[self.dof_ids["wrist_roll"]] = self.config.vel_limit_wrist
        dq = np.clip(dq, -dq_lim, dq_lim)

        alpha = self.config.smooth_dq * np.ones(self.model.nv)
        alpha[self.dof_ids["wrist_flex"]] = self.config.smooth_dq_wrist
        alpha[self.dof_ids["wrist_roll"]] = self.config.smooth_dq_wrist
        self.dq_filt = (1.0 - alpha) * self.dq_filt + alpha * dq

        mj.mj_integratePos(self.model, self.q_des, self.dq_filt, self.control_dt)
        self.q_des[self.robot_qpos_indices] = np.clip(
            self.q_des[self.robot_qpos_indices],
            self.j_lo[self.robot_qpos_indices],
            self.j_hi[self.robot_qpos_indices],
        )

        for joint_name in self.ARM_JOINTS:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        gidx = self.act_ids["gripper"]
        gdof = self.dof_ids["gripper"]
        self.data.ctrl[gidx] = np.clip(
            self.data.ctrl[gidx] + gripper_delta * self.control_dt,
            self.j_lo[gdof],
            self.j_hi[gdof],
        )
        self.q_des[gdof] = self.data.ctrl[gidx]

        for _ in range(self.n_physics_per_control):
            mj.mj_step(self.model, self.data)

    def reset_to_home_position(self) -> None:
        """Reset robot arm to home position at the start of each episode."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self.data.qpos[self.dof_ids["shoulder_pan"]] = 0.0
        self.data.qpos[self.dof_ids["shoulder_lift"]] = -0.3
        self.data.qpos[self.dof_ids["elbow_flex"]] = 0.6
        self.data.qpos[self.dof_ids["wrist_flex"]] = 1.2
        self.data.qpos[self.dof_ids["wrist_roll"]] = 0.0
        self.data.qpos[self.dof_ids["gripper"]] = self.config.home_gripper

        for joint_name in self.JOINT_NAMES:
            self.data.qvel[self.dof_ids[joint_name]] = 0.0

        self.q_des = self.data.qpos.copy()

        for joint_name in self.JOINT_NAMES:
            self.data.ctrl[self.act_ids[joint_name]] = self.q_des[self.dof_ids[joint_name]]

        self.dq_filt = np.zeros(self.model.nv)
        mj.mj_forward(self.model, self.data)

        ee_pos = self.data.site_xpos[self.ee_site_id]
        logger.info(
            f"Robot reset to home position - EE at: [{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]"
        )

    def _set_block_pose(
        self, x: float, y: float, z: float, yaw_deg: float
    ) -> tuple[float, float, float, float] | None:
        """Write a block pose into qpos, zero its velocity and refresh solver state.

        Returns the applied pose, or None if the scene has no block body.
        """
        block_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "block")
        if block_body_id < 0:
            logger.warning("Block body not found in model - skipping reset")
            return None

        block_jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "block")
        block_qpos_adr = self.model.jnt_qposadr[block_jnt_id]

        self.data.qpos[block_qpos_adr : block_qpos_adr + 3] = [x, y, z]

        yaw_rad = np.deg2rad(yaw_deg)
        quat = np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)])
        self.data.qpos[block_qpos_adr + 3 : block_qpos_adr + 7] = quat

        block_qvel_adr = self.model.jnt_dofadr[block_jnt_id]
        self.data.qvel[block_qvel_adr : block_qvel_adr + 6] = 0.0

        mj.mj_forward(self.model, self.data)

        logger.info(f"Block reset to position: [{x:.3f}, {y:.3f}, {z:.3f}], yaw: {yaw_deg}°")
        return (x, y, z, yaw_deg)

    def reset_block_position(self, episode_index: int) -> tuple[float, float, float, float] | None:
        """Set block position from predefined per-episode positions."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        if not self.cube_positions:
            raise ValueError(
                f"No cube positions loaded. Please ensure cube_positions.json exists at "
                f"{self.config.cube_positions_path}"
            )

        matching_pos = next((p for p in self.cube_positions if p.get("episode") == episode_index), None)
        if not matching_pos:
            raise ValueError(
                f"No predefined position found for episode {episode_index}. "
                f"Please add this episode to {self.config.cube_positions_path}"
            )

        x = float(matching_pos["x"])
        y = float(matching_pos["y"])
        z = float(matching_pos.get("z", 0.012))
        yaw_deg = float(matching_pos.get("yaw_deg", 0.0))

        logger.info(
            f"Using predefined position for episode {episode_index}: "
            f"[{x:.3f}, {y:.3f}, {z:.3f}], yaw: {yaw_deg}°"
        )
        return self._set_block_pose(x, y, z, yaw_deg)

    def reset_block_position_random(self) -> tuple[float, float, float, float] | None:
        """Sample a block pose uniformly within the configured graspable bounds."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        xr = self.config.cube_random_x_range
        yr = self.config.cube_random_y_range
        yawr = self.config.cube_random_yaw_range
        x = float(np.random.uniform(xr[0], xr[1]))
        y = float(np.random.uniform(yr[0], yr[1]))
        z = float(self.config.cube_random_z)
        yaw_deg = float(np.random.uniform(yawr[0], yawr[1]))

        logger.info(f"Random cube position: [{x:.3f}, {y:.3f}, {z:.3f}], yaw: {yaw_deg:.1f}°")
        return self._set_block_pose(x, y, z, yaw_deg)

    def _reset_arm_follow(self) -> None:
        """Keep the arm where it is (no teleport) for passive-leader recording.

        Zeroes residual joint velocity and holds the current pose via ctrl so the
        arm stays put until the leader teleop takes over on the next frame. This
        avoids the large first-frame jump that a fixed-home teleport would cause
        when the real leader is left wherever the operator's hand ended up.
        """
        for joint_name in self.JOINT_NAMES:
            self.data.qvel[self.dof_ids[joint_name]] = 0.0
            self.data.ctrl[self.act_ids[joint_name]] = self.data.qpos[self.dof_ids[joint_name]]
        self.q_des = self.data.qpos.copy()
        self.dq_filt = np.zeros(self.model.nv)
        mj.mj_forward(self.model, self.data)

    def reset_episode(self, episode_index: int) -> None:
        """Reset arm and/or block for a new recorded episode per config switches.

        reset_arm:  "home" teleports to fixed home; "follow" leaves the arm in place
                    (for the passive real leader arm).
        reset_cube: "fixed" uses cube_positions.json; "random" samples within bounds;
                    "none" leaves the cube untouched.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        reset_arm = getattr(self.config, "reset_arm", "home")
        reset_cube = getattr(self.config, "reset_cube", "fixed")

        if reset_arm == "follow":
            self._reset_arm_follow()
        else:
            self.reset_to_home_position()

        if reset_cube == "random":
            self.reset_block_position_random()
        elif reset_cube == "none":
            logger.info("Cube reset skipped (reset_cube=none)")
        else:
            self.reset_block_position(episode_index)

        settle_steps = max(1, int(0.5 / self.physics_dt))
        for _ in range(settle_steps):
            mj.mj_step(self.model, self.data)

        self._render_glfw()
        logger.info(f"Episode {episode_index} sim reset complete")

    def get_block_position(self) -> tuple[float, float, float] | None:
        """Get current block position for episode metadata."""
        if not self.is_connected:
            return None

        block_jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "block")
        if block_jnt_id < 0:
            return None

        block_qpos_adr = self.model.jnt_qposadr[block_jnt_id]
        pos = self.data.qpos[block_qpos_adr : block_qpos_adr + 3]
        return (float(pos[0]), float(pos[1]), float(pos[2]))

    def set_block_position_direct(self, x: float, y: float, z: float, yaw_deg: float = 0.0) -> None:
        """Directly set block position (for replay)."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        block_jnt_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, "block")
        if block_jnt_id < 0:
            logger.warning("Block not found in model - cannot set position")
            return

        block_qpos_adr = self.model.jnt_qposadr[block_jnt_id]
        self.data.qpos[block_qpos_adr : block_qpos_adr + 3] = [x, y, z]

        yaw_rad = np.deg2rad(yaw_deg)
        quat = np.array([np.cos(yaw_rad / 2), 0, 0, np.sin(yaw_rad / 2)])
        self.data.qpos[block_qpos_adr + 3 : block_qpos_adr + 7] = quat

        block_qvel_adr = self.model.jnt_dofadr[block_jnt_id]
        self.data.qvel[block_qvel_adr : block_qvel_adr + 6] = 0.0

        mj.mj_forward(self.model, self.data)
        logger.info(f"Block set to position: [{x:.3f}, {y:.3f}, {z:.3f}], yaw: {yaw_deg}°")

    def disconnect(self) -> None:
        """Close MuJoCo model and renderer."""
        if not self.is_connected:
            return

        # Suppress harmless GLFW "library not initialized" warnings that can be
        # emitted by underlying renderers/contexts during teardown.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=glfw.GLFWError)

            # glfwTerminate destroys all remaining windows/cursors, so we do not need
            # to call glfwDestroyWindow first.
            if self._glfw_initialized:
                self._glfw_initialized = False
                try:
                    glfw.terminate()
                except Exception as e:
                    logger.warning(f"Error terminating GLFW: {e}")
                self._glfw_window = None

            self._glfw_cam = None
            self._glfw_opt = None
            self._glfw_scene = None
            self._glfw_ctx = None

            if self._viewer is not None:
                try:
                    self._viewer.close()
                except Exception as e:
                    logger.warning(f"Error closing viewer: {e}")
                finally:
                    self._viewer = None

            for cam_name, renderer in list(self._renderers.items()):
                try:
                    renderer.close()
                except Exception as e:
                    logger.warning(f"Error closing renderer for {cam_name}: {e}")
            self._renderers.clear()

        self.model = None
        self.data = None
        self.q_des = None
        self.dq_filt = None

        logger.info(f"{self} disconnected")


# Alias expected by LeRobot's make_device_from_device_class factory.
SO101Mujoco = SO101MujocoRobot
