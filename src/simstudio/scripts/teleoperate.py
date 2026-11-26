"""Leader arm teleoperation: SO-101 leader arm → MuJoCo simulation.

Reads joint positions from Feetech STS3215 leader arm and sends them
as position targets to the MuJoCo SO-101 robot.

Usage:
    uv run python -m simstudio.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml
"""

import logging
import os
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
    if hasattr(teleop, "consume_reset_request"):
        print("Scene reset: Backspace (keyboard) or Plus (Joy-Con).")
    dt = 1.0 / cfg.fps
    last_reset_t = 0.0
    reset_debounce_s = 0.4

    try:
        while True:
            start = time.perf_counter()

            # Leader arm get_action() already outputs MuJoCo radians
            action = teleop.get_action()

            consume_reset = getattr(teleop, "consume_reset_request", None)
            if callable(consume_reset) and consume_reset():
                now = time.perf_counter()
                if now - last_reset_t >= reset_debounce_s:
                    last_reset_t = now
                    reset_fn = getattr(robot, "reset_teleop_scene", None)
                    if callable(reset_fn):
                        print("Resetting scene (arm / cube / drawer)...")
                        reset_fn(0)
                        on_reset = getattr(teleop, "on_scene_reset", None)
                        if callable(on_reset):
                            on_reset()
                        print("Scene reset done.")
                    else:
                        print("Reset requested, but this robot has no reset_teleop_scene().")
                # Skip applying this frame's action after a teleport.
                elapsed = time.perf_counter() - start
                if elapsed < dt:
                    time.sleep(dt - elapsed)
                continue

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
    # Prefer focus-independent keyboard on Wayland/desktop launches. pynput can
    # report "alive" yet capture no keys when the MuJoCo window has focus.
    # Same default as record.py; override with SO101_PREFER_EVDEV=0 if needed.
    os.environ.setdefault("SO101_PREFER_EVDEV", "1")
    teleoperate()


if __name__ == "__main__":
    main()
