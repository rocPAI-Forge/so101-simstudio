"""SO-101 keyboard teleoperator configuration."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_keyboard")
@dataclass
class SO101KeyboardTeleopConfig(TeleoperatorConfig):
    """Configuration for the SO-101 keyboard teleoperator."""

    lin_speed: float = 0.04
    yaw_speed: float = 1.20
    grip_speed: float = 0.7
