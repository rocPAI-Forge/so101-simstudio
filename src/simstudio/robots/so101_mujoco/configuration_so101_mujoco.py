"""SO-101 MuJoCo robot configuration."""

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_mujoco")
@dataclass
class SO101MujocoConfig(RobotConfig):
    """Configuration for the SO-101 MuJoCo simulation robot."""

    xml_path: Path = Path("SO101/scenes/simple_pick/scene.xml")
    cube_positions_path: Path | None = Path("configs/scenes/simple_pick/cube_positions.json")

    # Frequencies
    record_fps: int = 30
    control_fps: int = 180
    physics_fps: int = 360

    # Cameras
    camera_width: int = 640
    camera_height: int = 480
    camera_names: list[str] = field(default_factory=lambda: ["front", "top", "wrist"])
    show_collision_geometry: bool = False

    # Live GLFW rendering window (disable for headless/CI)
    render_window: bool = True

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

    # Action mode: "velocity" (keyboard) or "position" (leader arm)
    action_mode: str = "velocity"

    # Horizontal velocity interpretation for velocity teleop:
    #   "world"       -> vx/vy are world-frame X/Y (keyboard default).
    #   "cylindrical" -> vx = radial reach (extend/retract away from the base),
    #                    vy = tangential swing (base shoulder_pan arc). This is more
    #                    intuitive for a stick: forward/back extends, left/right swings.
    horizontal_control_mode: str = "world"
    # World XY of the base (shoulder_pan) pivot, used for the cylindrical transform.
    # Leave at [0.0, 0.0] to auto-read the true shoulder_pan anchor each step;
    # set a non-zero override only if you need to shift the pivot manually.
    base_xy: list[float] = field(default_factory=lambda: [0.0, 0.0])

    # Gripper joint value (rad) used at home/reset. MuJoCo gripper ctrlrange is
    # roughly (-0.17, 1.75): lower bound = fully closed, upper = fully open.
    # Leader arm starts fully closed, so leader configs set this to the closed bound.
    home_gripper: float = 0.8

    # Episode reset: "auto" resets arm + block before each recorded episode (sim);
    # "manual" leaves state unchanged (LeRobot real-robot style).
    reset_mode: str = "auto"

    # Fine-grained reset behaviour, applied only when reset_mode == "auto".
    #   reset_arm:  "home"   -> teleport arm to the fixed home pose (keyboard/replay).
    #               "follow" -> do NOT teleport; keep the arm where it is and let the
    #                           teleop take over next frame. Use this for the passive
    #                           real leader arm, otherwise the position mapping yanks the
    #                           sim arm from home to the leader's pose on the first frame.
    #   reset_cube: "fixed"  -> per-episode predefined position from cube_positions.json.
    #               "random" -> sample uniformly within the graspable bounds below.
    #               "none"   -> leave the cube untouched.
    reset_arm: str = "home"
    reset_cube: str = "fixed"

    # Random cube placement bounds (metres / degrees), used when reset_cube == "random".
    # Defaults cover the graspable rectangle observed in cube_positions.json.
    cube_random_x_range: list[float] = field(default_factory=lambda: [0.26, 0.34])
    cube_random_y_range: list[float] = field(default_factory=lambda: [0.165, 0.235])
    cube_random_z: float = 0.0125
    cube_random_yaw_range: list[float] = field(default_factory=lambda: [0.0, 0.0])
