"""Nintendo Switch Joy-Con teleoperator for SO-101 MuJoCo.

Translation reads the analog stick + up/down buttons directly from the Joy-Con HID
report and emits a constant velocity while deflected (bypassing the joycon-robotics
position integration, which couples axes via a first-person pointing vector). The
stick is mapped to an arm-centric scheme that pairs with the robot's
``horizontal_control_mode: cylindrical``:
- Stick forward/back -> vx : reach in/out (extend/retract the arm)
- Stick left/right   -> vy : base swing (shoulder_pan arc, left/right)
- R/L button / stick-press -> vz (+/- Z, up/down)
- Controller tilt (IMU roll/yaw) -> wrist_flex_rate / yaw_rate
- ZR/ZL hold -> close gripper, release -> open

(With the robot in the default ``world`` mode, vx/vy are world X/Y instead.)
Recording (episode) controls run through the project's focus-independent evdev
keyboard listener (see record.py), not Joy-Con buttons.
"""

import logging
import os
import time
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction

from simstudio.teleoperators.so101_joycon.config import SO101JoyConTeleopConfig

logger = logging.getLogger(__name__)

# Set SO101_JOYCON_DEBUG=1 to log raw stick values + computed velocities.
_DEBUG = os.environ.get("SO101_JOYCON_DEBUG") == "1"


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
        self._connect_time = 0.0
        self._last_debug_t = 0.0
        # Gripper toggle state (used when config.gripper_toggle is True).
        self._gripper_closed = False
        self._grip_prev_pressed = False

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
        # Initialize previous orientation state (used only for tilt-based rotation).
        posture, _, _ = self._jc.get_control()
        self._prev_posture = list(posture)
        self._prev_time = time.time()
        self._connect_time = self._prev_time
        print(f"✓ Connected to {side} Joy-Con")
        up_btn = "R" if side == "right" else "L"
        grip_btn = "ZR" if side == "right" else "ZL"
        print(f"  Stick: forward/back=reach in/out, left/right=base swing; {up_btn}=up, stick press=down")
        print("  Tilt controller: wrist rotation (roll=flex, yaw=wrist roll)")
        if self.config.gripper_toggle:
            print(f"  {grip_btn}: press to toggle gripper close/open (no need to hold)")
        else:
            print(f"  Hold {grip_btn}=close gripper, release=open gripper")
        # Recording (episode) controls are handled by the keyboard via the project's
        # focus-independent evdev listener — NOT by Joy-Con buttons. See record.py.
        print("  Recording controls (keyboard, focus-independent): N/Right save & next, R/Left re-record, Q/ESC stop")

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _stick_dir(self, raw: int) -> float:
        """Map a raw 12-bit stick axis to -1/0/+1 using a center + deadzone.

        The analog stick reads ~2047 at rest and swings to the full 0..4095 range at
        full deflection, so a value near 0 is a legitimate full-negative push (NOT an
        invalid read). Only values inside the deadzone around center produce no motion.
        """
        center = self.config.stick_center
        dead = self.config.stick_deadzone
        if raw > center + dead:
            return 1.0
        if raw < center - dead:
            return -1.0
        return 0.0

    def _condition_rate(self, rate: float) -> float:
        """Deadzone + clamp a tilt-derived rotation rate to suppress IMU drift/spikes."""
        if abs(rate) < self.config.rotation_deadzone:
            return 0.0
        rmax = self.config.rotation_max
        return max(-rmax, min(rmax, rate))

    def get_action(self) -> RobotAction:
        if not self._connected or self._jc is None:
            raise RuntimeError("Joy-Con not connected")

        jc = self._jc.joycon
        right = self.config.side == "right"

        # --- Translation: read the analog stick + up/down buttons directly ---
        # This is axis-aligned and fully decoupled (unlike the library's
        # orientation-dependent position integration).
        if right:
            stick_h = jc.get_stick_right_horizontal()
            stick_v = jc.get_stick_right_vertical()
            up_pressed = bool(jc.get_button_r())
            down_pressed = bool(jc.get_button_r_stick())
        else:
            stick_h = jc.get_stick_left_horizontal()
            stick_v = jc.get_stick_left_vertical()
            up_pressed = bool(jc.get_button_l())
            down_pressed = bool(jc.get_button_l_stick())

        xy_speed = self.config.lin_speed * self.config.translation_scale
        z_speed = self.config.lin_speed * self.config.z_scale

        # forward/back stick -> vx (reach), left/right stick -> vy (base swing)
        vx = self._stick_dir(stick_v) * xy_speed
        vy = self._stick_dir(stick_h) * xy_speed
        vz = (1.0 if up_pressed else (-1.0 if down_pressed else 0.0)) * z_speed

        if self.config.invert_x:
            vx = -vx
        if self.config.invert_y:
            vy = -vy
        if self.config.invert_z:
            vz = -vz

        # --- Rotation: derived from controller tilt (IMU), independent of sticks ---
        posture, _gripper_state, _button_control = self._jc.get_control()
        roll, yaw = posture[3], posture[5]

        current_time = time.time()
        dt = current_time - self._prev_time
        if dt < 0.001:
            dt = 0.001
        self._prev_time = current_time

        if self._prev_posture is not None:
            droll = (roll - self._prev_posture[3]) / dt
            dyaw = (yaw - self._prev_posture[5]) / dt
        else:
            droll = dyaw = 0.0
        self._prev_posture = list(posture)

        # Suppress rotation while the IMU settles right after connect.
        if (current_time - self._connect_time) < self.config.rotation_settle_s:
            wrist_flex_rate = 0.0
            yaw_rate = 0.0
        else:
            wrist_flex_rate = self._condition_rate(droll * self.config.rotation_scale)
            yaw_rate = self._condition_rate(dyaw * self.config.rotation_scale)

        # --- Gripper: ZR (right) or ZL (left) ---
        # toggle mode: each press flips closed/open (no holding needed);
        # hold mode: pressed = close, released = open.
        report = self._jc.joycon._input_report
        grip_pressed = bool((report[3] >> 7) & 1) if report[0] == 0x30 else False
        if self.config.gripper_toggle:
            if grip_pressed and not self._grip_prev_pressed:
                self._gripper_closed = not self._gripper_closed
            self._grip_prev_pressed = grip_pressed
            gripper_delta = -1.0 if self._gripper_closed else 1.0
        else:
            gripper_delta = -1.0 if grip_pressed else 1.0

        if _DEBUG and (current_time - self._last_debug_t) > 0.2:
            self._last_debug_t = current_time
            logger.info(
                "JOYCON stick_h=%d stick_v=%d up=%d down=%d -> vx=%.3f vy=%.3f vz=%.3f | "
                "roll=%.2f yaw=%.2f wf=%.3f yr=%.3f grip=%.0f",
                stick_h, stick_v, int(up_pressed), int(down_pressed),
                vx, vy, vz, roll, yaw, wrist_flex_rate, yaw_rate, gripper_delta,
            )

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
