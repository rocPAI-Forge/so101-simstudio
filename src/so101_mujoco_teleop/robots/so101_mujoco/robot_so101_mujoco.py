"""SO-101 MuJoCo robot implementation.

Adapted from robopicker/lerobot/src/lerobot/robots/so101_mujoco/robot_so101_mujoco.py
for latest HuggingFace LeRobot.
"""

from pathlib import Path
from typing import Any

import glfw
import mujoco as mj
import numpy as np
from lerobot.robots.robot import Robot
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.errors import DeviceNotConnectedError

from so101_mujoco_teleop.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig


class SO101MujocoRobot(Robot):
    """SO-101 robot in MuJoCo simulation."""

    config_class = SO101MujocoConfig
    name = "so101_mujoco"

    # TODO: port full implementation from robopicker
    # This is a skeleton to verify LeRobot plugin registration.

    def __init__(self, config: SO101MujocoConfig):
        super().__init__(config)
        self.config = config
        self.model: mj.MjModel | None = None
        self.data: mj.MjData | None = None

    @property
    def is_connected(self) -> bool:
        return self.model is not None and self.data is not None

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            return
        xml_path = Path(self.config.xml_path)
        if not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self.model = mj.MjModel.from_xml_path(str(xml_path))
        self.data = mj.MjData(self.model)

    def calibrate(self) -> None:
        pass

    def configure(self, **kwargs) -> None:
        pass

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot not connected")
        # TODO: implement full observation
        return {}

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError("Robot not connected")
        # TODO: implement Jacobian control and GLFW rendering
        return action

    def disconnect(self) -> None:
        self.model = None
        self.data = None
