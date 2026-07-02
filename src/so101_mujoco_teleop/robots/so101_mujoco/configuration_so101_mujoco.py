"""SO-101 MuJoCo robot configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs.robot import RobotConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MujocoConfig(RobotConfig):
    """Configuration for the SO-101 MuJoCo simulation robot."""

    xml_path: Path = Path("SO101/pick_scene.xml")
    cube_positions_path: Path | None = Path("configs/cube_positions.json")

    # Frequencies
    record_fps: int = 30
    control_fps: int = 180
    physics_fps: int = 360

    # Cameras
    camera_width: int = 640
    camera_height: int = 480
    camera_names: list[str] = field(default_factory=lambda: ["front", "top", "wrist"])
    show_collision_geometry: bool = True

    # Control speeds
    lin_speed: float = 0.04
    yaw_speed: float = 1.20
    grip_speed: float = 0.7

    # Orientation control
    ori_gain: float = 6.0
    tilt_deadzone: float = 0.03
    tilt_wmax: float = 6.0

    # Jacobian damping
    lambda_pos: float = 0.01
    lambda_tilt: float = 0.0001

    # Rate limiting / smoothing
    vel_limit: float = 0.5
    vel_limit_wrist: float = 8.0
    smooth_dq: float = 0.30
    smooth_dq_wrist: float = 0.08

    # Gravity compensation
    wrist_gff_gain: float = 0.5

    # Safety
    table_z: float = 0.0
    clearance: float = 0.07

    # End-effector
    ee_site_name: str = "wrist_site"
    tool_axis_site: list[float] = field(default_factory=lambda: [0.0, -1.0, 0.0])
