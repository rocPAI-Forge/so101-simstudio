"""SO-101 real leader arm teleoperator configuration."""

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("so101_leader_arm")
@dataclass
class SO101LeaderTeleopConfig(TeleoperatorConfig):
    """Configuration for using a real SO-101 leader arm as teleoperator input.

    Compatible with LeRobot's SOLeader implementation (Feetech STS3215).
    """

    port: str = "/dev/ttyACM0"
    use_degrees: bool = False
