"""SO-101 keyboard teleoperator.

Outputs normalized velocity commands for SO-101 end-effector control.

During **record** (``--view_mode mujoco`` or ``rerun``), SimStudio prefers evdev
when available so movement and episode controls work the same regardless of which
window has focus. Teleoperate-only sessions keep pynput-first behavior.
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

# Set by simstudio.scripts.record when keyboard recording patch is active.
_shared_recording_events: dict[str, bool] | None = None
# Last connect() backend: "evdev", "pynput", or None (record.py reads this).
_keyboard_input_backend: str | None = None
# False while save_episode / gaps between record_loop iterations (evdev still runs).
_recording_keys_enabled: bool = False


def set_recording_keys_enabled(enabled: bool) -> None:
    """Enable/disable recording hotkeys outside active ``record_loop`` iterations."""
    global _recording_keys_enabled
    _recording_keys_enabled = enabled

EVDEV_AVAILABLE = False
try:
    from simstudio.teleoperators.so101_keyboard.evdev_listener import EvdevKeyListener

    EVDEV_AVAILABLE = True
except ImportError:
    EvdevKeyListener = None  # type: ignore[misc, assignment]

PYNPUT_AVAILABLE = False
keyboard = None
try:
    from lerobot.utils.import_utils import _pynput_available

    PYNPUT_AVAILABLE = _pynput_available
    if PYNPUT_AVAILABLE:
        from pynput import keyboard
except Exception as e:
    logger.info("Could not import pynput keyboard backend: %s", e)


class SO101KeyboardTeleop(Teleoperator):
    """Keyboard teleoperator for SO-101.

    Movement keys (hold to move):
      W/S : +Y / -Y
      A/D : -X / +X
      Z/X : +Z / -Z
      I/K : wrist flex up/down
      [/] : wrist roll left/right
      O/C : gripper open/close

    Recording control keys (press once; same in mujoco and rerun record modes):
      Right arrow / N : save current episode, move to next
      Left arrow  / R : cancel current episode, rerecord
      ESC         / Q : stop recording entirely
    """

    name = "so101_keyboard"
    config_class = SO101KeyboardTeleopConfig

    def __init__(self, config: SO101KeyboardTeleopConfig):
        if not EVDEV_AVAILABLE:
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
        return (
            PYNPUT_AVAILABLE
            and isinstance(self.listener, keyboard.Listener)
            and self.listener.is_alive()
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        global _keyboard_input_backend

        self._use_evdev = False
        _keyboard_input_backend = None
        prefer_evdev = os.environ.get("SO101_PREFER_EVDEV") == "1"

        def _try_evdev() -> bool:
            if not EVDEV_AVAILABLE or EvdevKeyListener is None:
                return False
            try:
                evdev_listener = EvdevKeyListener(on_key=self._on_evdev_key)
                if evdev_listener.start():
                    self._evdev_listener = evdev_listener
                    self._use_evdev = True
                    logger.info("Using evdev keyboard listener (focus-independent).")
                    return True
            except Exception as e:
                logger.warning("evdev listener failed: %s", e)
            return False

        def _try_pynput() -> bool:
            if not PYNPUT_AVAILABLE:
                return False
            try:
                self.listener = keyboard.Listener(
                    on_press=self._on_press,
                    on_release=self._on_release,
                )
                self.listener.start()
                time.sleep(0.1)
                if self.listener.is_alive():
                    logger.info("Using pynput keyboard listener.")
                    return True
                logger.warning(
                    "pynput listener started but is not alive. Keyboard teleoperator will produce no actions."
                )
                self.listener = None
            except Exception as e:
                logger.warning("Failed to start pynput keyboard listener: %s", e)
                self.listener = None
            return False

        if prefer_evdev:
            if not _try_evdev():
                _try_pynput()
        else:
            if not _try_pynput():
                _try_evdev()

        if not self.is_connected:
            logger.warning(
                "Neither pynput nor evdev available. Keyboard teleoperator will produce no actions."
            )
            _keyboard_input_backend = None
        elif self._use_evdev:
            _keyboard_input_backend = "evdev"
        else:
            _keyboard_input_backend = "pynput"

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _handle_recording_key(self, key_name: str, is_pressed: bool) -> bool:
        """Apply recording-control keys. Returns True if the event was consumed."""
        if not is_pressed or not _recording_keys_enabled:
            return False
        if self._recording_events is None and _shared_recording_events is not None:
            self.set_recording_events(_shared_recording_events)
        if self._recording_events is None:
            return False

        from simstudio.common.recording_controls import apply_keyboard_recording_key

        if apply_keyboard_recording_key(key_name, self._recording_events):
            return True
        return False

    def _on_evdev_key(self, key_name: str, is_pressed: bool) -> None:
        if self._handle_recording_key(key_name, is_pressed):
            return
        self.event_queue.put((key_name, is_pressed))

    def _on_press(self, key):
        if key == keyboard.Key.esc:
            self._handle_recording_key("esc", True)
            return
        if key == keyboard.Key.right:
            self._handle_recording_key("right", True)
            return
        if key == keyboard.Key.left:
            self._handle_recording_key("left", True)
            return

        key_char = getattr(key, "char", None)
        if key_char is not None:
            char = key_char.lower()
            if self._handle_recording_key(char, True):
                return
            self.event_queue.put((char, True))
        else:
            self.event_queue.put((str(key), True))

    def _on_release(self, key):
        key_char = getattr(key, "char", None)
        if key_char is not None:
            self.event_queue.put((key_char.lower(), False))
        else:
            self.event_queue.put((str(key), False))

    def _drain_pressed_keys(self):
        """Update current_pressed state from event queue."""
        while not self.event_queue.empty():
            key_char, is_pressed = self.event_queue.get_nowait()
            if is_pressed:
                self.current_pressed[key_char] = True
            else:
                self.current_pressed.pop(key_char, None)

    def get_action(self) -> RobotAction:
        """Return normalized velocity commands from currently pressed keys."""
        if self._recording_events is None and _shared_recording_events is not None:
            self.set_recording_events(_shared_recording_events)

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

    def disconnect(self) -> None:
        global _keyboard_input_backend

        if self._use_evdev and self._evdev_listener is not None:
            try:
                self._evdev_listener.stop()
            except Exception as e:
                logger.warning("Error stopping evdev listener: %s", e)
            finally:
                self._evdev_listener = None
        elif self.listener is not None:
            try:
                self.listener.stop()
            except Exception as e:
                logger.warning("Error stopping keyboard listener: %s", e)
            finally:
                self.listener = None
        self.current_pressed.clear()
        self._use_evdev = False
        _keyboard_input_backend = None
