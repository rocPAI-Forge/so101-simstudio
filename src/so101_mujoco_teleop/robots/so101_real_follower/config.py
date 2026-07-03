"""SO-101 real follower arm configuration (stub)."""

from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("so101_real_follower")
@dataclass
class SO101RealFollowerConfig(RobotConfig):
    """Configuration for the real SO-101 follower arm."""

    port: str = "/dev/ttyACM1"
    baudrate: int = 1000000
