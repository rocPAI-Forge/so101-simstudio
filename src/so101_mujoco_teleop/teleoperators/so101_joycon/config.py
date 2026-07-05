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

    # Button mapping (right Joy-Con defaults)
    gripper_button: str = "zr"  # ZR (right) or ZL (left) - auto-detected based on side
    next_episode_button: str = "a"  # Right: A, Left: Left arrow
    restart_episode_button: str = "y"  # Right: Y, Left: Up arrow
    stop_button: str = "plus"  # Right: Plus, Left: Minus
