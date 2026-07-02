"""SO-101 keyboard teleoperator.

Outputs normalized velocity commands for SO-101 end-effector control.
Uses pynput for global key capture, following the pattern in
lerobot.teleoperators.keyboard.teleop_keyboard.
"""

import logging
import time
from queue import Queue
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.import_utils import _pynput_available, require_package
from lerobot.utils.keyboard_input import pynput_can_capture

from so101_mujoco_teleop.teleoperators.so101_keyboard.config import SO101KeyboardTeleopConfig

logger = logging.getLogger(__name__)

PYNPUT_AVAILABLE = _pynput_available
keyboard = None
if PYNPUT_AVAILABLE:
    try:
        from pynput import keyboard
    except Exception as e:
        PYNPUT_AVAILABLE = False
        logger.info("Could not import pynput keyboard backend: %s", e)


class SO101KeyboardTeleop(Teleoperator):
    """Keyboard teleoperator for SO-101.

    Maps keys to normalized velocity commands:
      W/S : +Y / -Y
      A/D : -X / +X
      Q/E : +Z / -Z
      I/K : wrist flex up/down
      [/] : wrist roll left/right
      O/C : gripper open/close
    """

    name = "so101_keyboard"
    config_class = SO101KeyboardTeleopConfig

    def __init__(self, config: SO101KeyboardTeleopConfig):
        require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.event_queue: Queue[tuple[str, bool]] = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None
        self.logs: dict[str, Any] = {}

    @property
    def action_features(self) -> dict[str, type]:
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
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self) -> None:
        if PYNPUT_AVAILABLE and pynput_can_capture():
            logger.info("pynput is available - enabling local keyboard listener for SO-101 teleop.")
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
        else:
            logger.warning(
                "Keyboard teleoperation is unavailable in this environment. pynput can only "
                "capture key events on an X11 session (Linux), a Windows desktop, or macOS with "
                "Accessibility / Input Monitoring granted - not on Wayland or headless machines. "
                "This keyboard teleoperator will produce no actions."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _on_press(self, key):
        key_char = getattr(key, "char", None)
        if key_char is not None:
            self.event_queue.put((key_char.lower(), True))
        else:
            self.event_queue.put((str(key), True))

    def _on_release(self, key):
        key_char = getattr(key, "char", None)
        if key_char is not None:
            self.event_queue.put((key_char.lower(), False))
        else:
            self.event_queue.put((str(key), False))

        if key == keyboard.Key.esc:
            logger.info("ESC pressed, disconnecting SO-101 keyboard teleop.")
            self.disconnect()

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue."""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                self.current_pressed.pop(key_char, None)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Return normalized velocity commands from currently pressed keys."""
        before_read_t = time.perf_counter()
        self._drain_pressed_keys()

        # Build set of active keys from current_pressed (only True entries remain)
        active_keys = {key for key, is_pressed in self.current_pressed.items() if is_pressed}

        vx = 0.0
        vy = 0.0
        vz = 0.0
        wrist_flex_rate = 0.0
        yaw_rate = 0.0
        gripper_delta = 0.0

        if "w" in active_keys:
            vy += self.config.lin_speed
        if "s" in active_keys:
            vy -= self.config.lin_speed
        if "a" in active_keys:
            vx -= self.config.lin_speed
        if "d" in active_keys:
            vx += self.config.lin_speed
        if "q" in active_keys:
            vz += self.config.lin_speed
        if "e" in active_keys:
            vz -= self.config.lin_speed

        if "i" in active_keys:
            wrist_flex_rate -= self.config.yaw_speed
        if "k" in active_keys:
            wrist_flex_rate += self.config.yaw_speed

        if "[" in active_keys:
            yaw_rate -= self.config.yaw_speed
        if "]" in active_keys:
            yaw_rate += self.config.yaw_speed

        if "o" in active_keys:
            gripper_delta += self.config.grip_speed
        if "c" in active_keys:
            gripper_delta -= self.config.grip_speed

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return {
            "vx": vx,
            "vy": vy,
            "vz": vz,
            "wrist_flex_rate": wrist_flex_rate,
            "yaw_rate": yaw_rate,
            "gripper_delta": gripper_delta,
        }

    def send_feedback(self, action: RobotAction, **kwargs) -> None:
        pass

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.listener is not None:
            try:
                self.listener.stop()
            except Exception as e:
                logger.warning(f"Error stopping keyboard listener: {e}")
            finally:
                self.listener = None
        self.current_pressed.clear()
