"""SO-101 real leader arm configuration (stub)."""

from dataclasses import dataclass

from lerobot.configs.robot import RobotConfig


@RobotConfig.register_subclass("so101_leader")
@dataclass
class SO101LeaderConfig(RobotConfig):
    """Configuration for the real SO-101 leader arm."""

    port: str = "/dev/ttyACM0"
    baudrate: int = 1000000
