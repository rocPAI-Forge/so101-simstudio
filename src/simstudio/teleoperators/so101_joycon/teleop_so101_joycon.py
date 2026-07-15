"""Nintendo Switch Joy-Con teleoperator for SO-101 MuJoCo.

Uses joycon-robotics library's position-based control, converted to velocity commands.
This matches the reference project's intuitive control scheme:
- Stick accumulates position offset → converted to velocity
- Gyroscope accumulates orientation → converted to rotation rate
- Buttons control gripper and recording events
"""

import time
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from simstudio.teleoperators.so101_joycon.config import SO101JoyConTeleopConfig


class SO101JoyConTeleop(Teleoperator):
    """Joy-Con teleoperator using joycon-robotics library's position control."""

    name = "so101_joycon"
    config_class = SO101JoyConTeleopConfig

    def __init__(self, config: SO101JoyConTeleopConfig):
        super().__init__(config)
        self.config = config
        self._connected = False
        self._jc = None  # JoyconRobotics instance
        self._prev_posture = None  # [x, y, z, roll, pitch, yaw]
        self._prev_time = None

    @property
    def action_features(self) -> dict:
        return {
            "vx": float,
            "vy": float,
            "vz": float,
            "wrist_flex_rate": float,
            "yaw_rate": float,
            "gripper_delta": float,
        }

    @property
    def feedback_features(self) -> dict:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_calibrated(self) -> bool:
        return True

    def connect(self, calibrate: bool = False) -> None:
        try:
            from joyconrobotics import JoyconRobotics
        except ImportError as err:
            raise ImportError("joyconrobotics not installed") from err

        side = self.config.side
        print(f"Connecting to {side} Joy-Con...")
        self._jc = JoyconRobotics(
            side,
            without_rest_init=True,
            common_rad=True,
            lerobot=False,
            pure_z=True,
            pure_dx=True,
            change_down_to_gripper=False,
        )
        self._connected = True
        # Initialize previous state
        posture, _, _ = self._jc.get_control()
        self._prev_posture = list(posture)
        self._prev_time = time.time()
        print(f"✓ Connected to {side} Joy-Con")
        if side == "right":
            print("  Stick: left/right=X axis, forward/back=Y axis; R=up, stick press=down")
            print("  Tilt controller: wrist rotation")
            print("  Hold ZR=close gripper, release=open gripper")
        else:
            print("  Stick: left/right=X axis, forward/back=Y axis; L=up, stick press=down")
            print("  Tilt controller: wrist rotation")
            print("  Hold ZL=close gripper, release=open gripper")
        # Recording (episode) controls are handled by the keyboard via the project's
        # focus-independent evdev listener — NOT by Joy-Con buttons. See record.py.
        print("  Recording controls (keyboard, focus-independent): N/Right save & next, R/Left re-record, Q/ESC stop")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> RobotAction:
        if not self._connected or self._jc is None:
            raise RuntimeError("Joy-Con not connected")

        # Get position and orientation from joycon-robotics.
        # button_control is intentionally ignored: recording (episode) controls run
        # through the keyboard evdev listener (see record.py), not Joy-Con buttons.
        posture, gripper_state, _button_control = self._jc.get_control()
        # posture = [x, y, z, roll, pitch, yaw]
        x, y, z, roll, pitch, yaw = posture

        # Calculate dt
        current_time = time.time()
        dt = current_time - self._prev_time
        if dt < 0.001:
            dt = 0.001
        self._prev_time = current_time

        # Convert position difference to velocity
        if self._prev_posture is not None:
            dx = (x - self._prev_posture[0]) / dt
            dy = (y - self._prev_posture[1]) / dt
            dz = (z - self._prev_posture[2]) / dt
            droll = (roll - self._prev_posture[3]) / dt
            dyaw = (yaw - self._prev_posture[5]) / dt
        else:
            dx = dy = dz = droll = dyaw = 0.0

        self._prev_posture = [x, y, z, roll, pitch, yaw]

        # Scale velocities
        vx = dx * self.config.translation_scale
        vy = dy * self.config.translation_scale
        vz = dz * self.config.z_scale

        # Gyroscope rotation
        wrist_flex_rate = droll * self.config.rotation_scale
        yaw_rate = dyaw * self.config.rotation_scale

        # Gripper: ZR (right) or ZL (left) hold = close, release = open
        # gripper_delta: negative = close, positive = open
        report = self._jc.joycon._input_report
        if self.config.side == "right":
            # ZR button: bit 7 of report[3]
            grip_pressed = bool((report[3] >> 7) & 1) if report[0] == 0x30 else False
        else:
            # ZL button: bit 7 of report[3]
            grip_pressed = bool((report[3] >> 7) & 1) if report[0] == 0x30 else False
        gripper_delta = -1.0 if grip_pressed else 1.0

        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "wrist_flex_rate": wrist_flex_rate,
            "yaw_rate": yaw_rate,
            "gripper_delta": gripper_delta,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass

    def disconnect(self) -> None:
        if self._jc is not None:
            try:
                self._jc.disconnnect()
            except Exception:
                pass
            self._jc = None
        self._connected = False
        print("Joy-Con disconnected")
