"""Smoke tests for plugin registration and imports."""


def test_robot_config_registration():
    from so101_mujoco_teleop.robots.so101_mujoco.configuration_so101_mujoco import (
        SO101MujocoConfig,
    )

    assert SO101MujocoConfig.name == "so101_mujoco"


def test_teleop_config_registration():
    from so101_mujoco_teleop.teleoperators.so101_keyboard.config import (
        SO101KeyboardTeleopConfig,
    )

    assert SO101KeyboardTeleopConfig.name == "so101_keyboard"
