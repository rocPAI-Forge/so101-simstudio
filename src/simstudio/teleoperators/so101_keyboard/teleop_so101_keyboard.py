"""SO-101 keyboard teleoperator.

Outputs normalized velocity commands for SO-101 end-effector control.
Uses evdev for Wayland (reads /dev/input/event* directly) or pynput for X11.
"""

import logging
import os
import time
from queue import Queue
from typing import Any

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.types import RobotAction
from lerobot.utils.decorators import check_if_already_connected

from simstudio.teleoperators.so101_keyboard.config import SO101KeyboardTeleopConfig

logger = logging.getLogger(__name__)

# Try evdev first (works on Wayland)
EVDEV_AVAILABLE = False
_evdev_listener = None
try:
    from simstudio.teleoperators.so101_keyboard.evdev_listener import EvdevKeyListener
    EVDEV_AVAILABLE = True
except ImportError:
    pass

# Try pynput as fallback (X11 only)
PYNPUT_AVAILABLE = False
keyboard = None
try:
    from lerobot.utils.import_utils import _pynput_available
    PYNPUT_AVAILABLE = _pynput_available
    if PYNPUT_AVAILABLE:
        from pynput import keyboard
except Exception:
    pass


class SO101KeyboardTeleop(Teleoperator):
    """Keyboard teleoperator for SO-101.

    Movement keys (hold to move):
      W/S : +Y / -Y
      A/D : -X / +X
      Z/X : +Z / -Z
      I/K : wrist flex up/down
      [/] : wrist roll left/right
      O/C : gripper open/close

    Recording control keys (press once):
      Right arrow : save current episode, move to next
      Left arrow  : cancel current episode, rerecord
      ESC         : stop recording entirely
    """

    name = "so101_keyboard"
    config_class = SO101KeyboardTeleopConfig

    def __init__(self, config: SO101KeyboardTeleopConfig):
        if EVDEV_AVAILABLE:
            pass  # evdev doesn't require pynput
        else:
            from lerobot.utils.import_utils import require_package
            require_package("pynput", extra="pynput-dep")
        super().__init__(config)
        self.config = config
        self.event_queue: Queue[tuple[str, bool]] = Queue()
        self.current_pressed: dict[str, bool] = {}
        self.listener = None
        self._evdev_listener = None
        self._use_evdev = False
        self.logs: dict[str, Any] = {}
        self._recording_events: dict[str, bool] | None = None
        # Low-pass filter for smoother velocity control
        self._prev_action: dict[str, float] = {}
        self._filter_alpha: float = 0.4  # 0=full smooth, 1=no smooth

    def set_recording_events(self, events: dict[str, bool]) -> None:
        """Set the recording events dict so arrow keys/ESC can control recording."""
        self._recording_events = events

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
        if self._use_evdev:
            return self._evdev_listener is not None and self._evdev_listener._running
        return PYNPUT_AVAILABLE and isinstance(self.listener, keyboard.Listener) and self.listener.is_alive()

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self._use_evdev = False

        # Try evdev first (works on Wayland, reads /dev/input/event* directly)
        if EVDEV_AVAILABLE:
            try:
                evdev_listener = EvdevKeyListener(on_key=self._on_evdev_key)
                if evdev_listener.start():
                    self._evdev_listener = evdev_listener
                    self._use_evdev = True
                    logger.info("Using evdev keyboard listener (Wayland-compatible)")
                    return
            except Exception as e:
                logger.warning(f"evdev listener failed: {e}")

        # Fallback to pynput (X11 only)
        if not PYNPUT_AVAILABLE:
            logger.warning(
                "Neither evdev nor pynput available. Keyboard teleoperator will produce no actions."
            )
            self.listener = None
            return

        # Try pynput directly
        try:
            self.listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
            self.listener.start()
            time.sleep(0.1)
            if self.listener.is_alive():
                logger.info("pynput keyboard listener started successfully (X11 mode)")
            else:
                logger.warning(
                    "pynput listener started but is not alive. Keyboard teleoperator will produce no actions."
                )
                self.listener = None
        except Exception as e:
            logger.warning(
                f"Failed to start pynput keyboard listener: {e}. "
                "Keyboard teleoperator will produce no actions."
            )
            self.listener = None

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _on_press(self, key):
        # Recording control: arrow keys and ESC (press once, not held)
        if key == keyboard.Key.esc:
            if self._recording_events is not None:
                print("Escape key pressed. Stopping data recording...")
                self._recording_events["stop_recording"] = True
                self._recording_events["exit_early"] = True
            return
        if key == keyboard.Key.right:
            if self._recording_events is not None:
                print("Right arrow pressed. Saving current episode...")
                self._recording_events["exit_early"] = True
            return
        if key == keyboard.Key.left:
            if self._recording_events is not None:
                print("Left arrow pressed. Canceling current episode...")
                self._recording_events["rerecord_episode"] = True
                self._recording_events["exit_early"] = True
            return

        # Movement keys (held)
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

    def _on_evdev_key(self, key_name: str, is_pressed: bool) -> None:
        """Handle evdev key events (called from evdev listener thread)."""
        # Recording control keys (press once, not held)
        if key_name == "esc":
            if self._recording_events is not None and is_pressed:
                print("Escape key pressed. Stopping data recording...")
                self._recording_events["stop_recording"] = True
                self._recording_events["exit_early"] = True
            return
        if key_name == "right":
            if self._recording_events is not None and is_pressed:
                print("Right arrow pressed. Saving current episode...")
                self._recording_events["exit_early"] = True
            return
        if key_name == "left":
            if self._recording_events is not None and is_pressed:
                print("Left arrow pressed. Canceling current episode...")
                self._recording_events["rerecord_episode"] = True
                self._recording_events["exit_early"] = True
            return

        # Movement keys (held)
        self.event_queue.put((key_name, is_pressed))

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue."""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                self.current_pressed.pop(key_char, None)

    def get_action(self) -> RobotAction:
        """Return normalized velocity commands from currently pressed keys.

        If pynput is unavailable (e.g. headless/Wayland), returns zero actions
        instead of raising an error, so the recording loop can continue.
        """
        before_read_t = time.perf_counter()

        if not self.is_connected:
            self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t
            return {
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "wrist_flex_rate": 0.0,
                "yaw_rate": 0.0,
                "gripper_delta": 0.0,
            }

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
        if "z" in active_keys:
            vz += self.config.lin_speed
        if "x" in active_keys:
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

        # Apply low-pass filter for smoother motion
        raw = {
            "vx": vx, "vy": vy, "vz": vz,
            "wrist_flex_rate": wrist_flex_rate,
            "yaw_rate": yaw_rate, "gripper_delta": gripper_delta,
        }
        if self._prev_action:
            for k in raw:
                raw[k] = self._filter_alpha * raw[k] + (1 - self._filter_alpha) * self._prev_action.get(k, 0.0)
        self._prev_action = raw.copy()

        return raw

    def send_feedback(self, action: RobotAction, **kwargs) -> None:
        pass

    def disconnect(self) -> None:
        if self._use_evdev and self._evdev_listener is not None:
            try:
                self._evdev_listener.stop()
            except Exception as e:
                logger.warning(f"Error stopping evdev listener: {e}")
            finally:
                self._evdev_listener = None
        elif self.listener is not None:
            try:
                self.listener.stop()
            except Exception as e:
                logger.warning(f"Error stopping keyboard listener: {e}")
            finally:
                self.listener = None
        self.current_pressed.clear()
