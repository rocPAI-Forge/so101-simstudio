"""SO-101 real leader arm teleoperator configuration (stub)."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_arm")
@dataclass
class SO101LeaderTeleopConfig(TeleoperatorConfig):
    """Configuration for using a real SO-101 leader arm as teleoperator input."""

    port: str = "/dev/ttyACM0"
    baudrate: int = 1000000
