"""Leader arm teleoperation: SO-101 leader arm → MuJoCo simulation.

Reads joint positions from Feetech STS3215 leader arm and sends them
as position targets to the MuJoCo SO-101 robot.

Usage:
    uv run python -m simstudio.scripts.teleoperate --config configs/so101_mujoco_leader_teleop.yaml
"""

import logging
import time
from dataclasses import dataclass, field

import rerun as rr
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
from simstudio.teleoperators.so101_leader import (  # noqa: F401
    SO101LeaderTeleopConfig,
)

RERUN_APPLICATION_ID = "simstudio"


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig = field(default_factory=TeleoperatorConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    fps: int = 30
    view_mode: str = "rerun"  # "mujoco", "rerun", or "both"


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    init_logging()
    logging.info("Teleoperate config loaded")

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    # Configure render_window based on view_mode
    if cfg.view_mode in ("mujoco", "both"):
        robot.config.render_window = True
    else:
        robot.config.render_window = False

    print("Connecting leader arm...")
    teleop.connect(calibrate=True)

    print("Connecting MuJoCo robot...")
    robot.connect(calibrate=False)

    # Initialize rerun for camera feed display
    if cfg.view_mode in ("rerun", "both"):
        rr.init(RERUN_APPLICATION_ID, spawn=True)

    print(f"Teleoperation started (view_mode={cfg.view_mode}). Press Ctrl+C to stop.")
    dt = 1.0 / cfg.fps
    frame_count = 0

    try:
        while True:
            start = time.perf_counter()

            # Leader arm get_action() already outputs MuJoCo radians
            action = teleop.get_action()

            # Send to MuJoCo robot (position-based)
            robot.send_action(action)

            # Log camera feeds to rerun every frame
            if cfg.view_mode in ("rerun", "both"):
                obs = robot.get_observation()
                rr.set_time("timeline", timestamp=time.time())
                for cam_name in robot.config.camera_names:
                    key = f"camera_{cam_name}"
                    if key in obs:
                        rr.log(key, rr.Image(obs[key]))

            frame_count += 1
            if frame_count % 100 == 0:
                elapsed = time.perf_counter() - start
                logging.info(f"Frame {frame_count} @ {1.0 / elapsed:.1f} FPS")

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
