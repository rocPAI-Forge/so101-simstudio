"""Smoke tests for plugin registration and imports."""

from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig


def test_robot_config_registration():
    from simstudio.robots.so101_mujoco.configuration_so101_mujoco import (
        SO101MujocoConfig,
    )

    assert RobotConfig.get_choice_name(SO101MujocoConfig) == "so101_mujoco"


def test_teleop_config_registration():
    from simstudio.teleoperators.so101_keyboard.config import (
        SO101KeyboardTeleopConfig,
    )

    assert TeleoperatorConfig.get_choice_name(SO101KeyboardTeleopConfig) == "so101_keyboard"


def test_robot_factory():
    from lerobot.robots import make_robot_from_config

    from simstudio.robots.so101_mujoco import SO101MujocoConfig

    robot = make_robot_from_config(SO101MujocoConfig())
    assert robot.name == "so101_mujoco"
    assert robot.action_features
    assert robot.observation_features


def test_teleop_factory():
    from lerobot.teleoperators import make_teleoperator_from_config

    from simstudio.teleoperators.so101_keyboard import SO101KeyboardTeleopConfig

    teleop = make_teleoperator_from_config(SO101KeyboardTeleopConfig())
    assert teleop.name == "so101_keyboard"
    assert teleop.action_features
