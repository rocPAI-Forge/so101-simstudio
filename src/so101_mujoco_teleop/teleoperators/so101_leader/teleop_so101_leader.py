"""SO-101 real leader arm teleoperator (stub)."""

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from so101_mujoco_teleop.teleoperators.so101_leader.config import SO101LeaderTeleopConfig


class SO101LeaderTeleop(Teleoperator):
    """Real SO-101 leader arm as teleoperator input (placeholder).

    Reads leader arm joint positions and uses them to drive a robot.
    """

    name = "so101_leader_arm"
    config_class = SO101LeaderTeleopConfig

    def __init__(self, config: SO101LeaderTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False

    def connect(self) -> None:
        raise NotImplementedError("so101_leader teleop not yet implemented")

    def is_connected(self) -> bool:
        return self._connected

    def get_action(self) -> RobotAction:
        raise NotImplementedError("so101_leader teleop not yet implemented")

    def send_feedback(self, action: RobotAction, **kwargs) -> None:
        pass

    def disconnect(self) -> None:
        self._connected = False
