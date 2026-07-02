"""Nintendo Switch Joy-Con teleoperator configuration (stub)."""

from dataclasses import dataclass

from lerobot.configs.teleoperator import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_joycon")
@dataclass
class SO101JoyConTeleopConfig(TeleoperatorConfig):
    """Configuration for the SO-101 Joy-Con teleoperator."""

    side: str = "left"  # left | right | both
    device: str = "auto"
