"""Nintendo Switch Joy-Con teleoperator configuration."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_joycon")
@dataclass
class SO101JoyConTeleopConfig(TeleoperatorConfig):
    """Configuration for the SO-101 Joy-Con teleoperator."""

    side: str = "right"  # left | right
    device: str = "auto"  # auto | left | right

    # Velocity scaling
    translation_scale: float = 2.0  # Scale for x, y translation
    z_scale: float = 1.0  # Scale for z translation
    rotation_scale: float = 0.5  # Scale for gyroscope rotation rates

    # Button mapping
    gripper_toggle_button: str = "zr"  # Button to toggle gripper
    next_episode_button: str = "a"  # Button for next episode
    restart_episode_button: str = "y"  # Button for restart episode
    reset_joycon_button: str = "plus"  # Button to reset Joy-Con calibration
