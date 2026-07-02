"""SO-101 real leader arm implementation (stub)."""

from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError

from so101_mujoco_teleop.robots.so101_leader.configuration_so101_leader import SO101LeaderConfig


class SO101LeaderRobot(Robot):
    """Real SO-101 leader arm (placeholder)."""

    config_class = SO101LeaderConfig
    name = "so101_leader"

    def __init__(self, config: SO101LeaderConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return False

    def connect(self, calibrate: bool = True) -> None:
        raise NotImplementedError("so101_leader hardware support not yet implemented")

    def calibrate(self) -> None:
        raise NotImplementedError("so101_leader hardware support not yet implemented")

    def configure(self, **kwargs) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        raise DeviceNotConnectedError("Robot not connected")

    def send_action(self, action: RobotAction) -> RobotAction:
        raise DeviceNotConnectedError("Robot not connected")

    def disconnect(self) -> None:
        self._connected = False
