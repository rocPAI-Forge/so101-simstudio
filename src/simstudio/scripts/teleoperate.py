"""Leader arm teleoperation: SO-101 leader arm → MuJoCo simulation.

Reads joint positions from Feetech STS3215 leader arm and sends them
as position targets to the MuJoCo SO-101 robot.

Usage:
    uv run python -m simstudio.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml
"""

import logging
import time
from dataclasses import dataclass, field

from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config
from lerobot.teleoperators import (
    TeleoperatorConfig,
    make_teleoperator_from_config,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

# ---------------------------------------------------------------------------
# 1. Register project plugins with LeRobot's ChoiceRegistry.
# ---------------------------------------------------------------------------
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from simstudio.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401
from simstudio.teleoperators.so101_keyboard import SO101KeyboardTeleopConfig  # noqa: F401
from simstudio.teleoperators.so101_leader import SO101LeaderTeleopConfig  # noqa: F401


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig = field(default_factory=TeleoperatorConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    fps: int = 30


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info("Teleoperate config loaded")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    print("Connecting leader arm...")
    teleop.connect(calibrate=True)

    print("Connecting MuJoCo robot...")
    robot.connect(calibrate=False)

    print("Teleoperation started. Press Ctrl+C to stop.")
    dt = 1.0 / cfg.fps

    try:
        while True:
            start = time.perf_counter()

            # Leader arm get_action() already outputs MuJoCo radians
            action = teleop.get_action()

            # Send to MuJoCo robot (position-based)
            robot.send_action(action)

            elapsed = time.perf_counter() - start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    except KeyboardInterrupt:
        print("\nStopping teleoperation...")
    finally:
        teleop.disconnect()
        robot.disconnect()
        print("Disconnected.")


def main():
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
