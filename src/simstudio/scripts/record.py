"""Project wrapper around LeRobot's record script.

This module imports project-specific robot and teleoperator plugins so they
are registered in LeRobot's config registries, then delegates to LeRobot's
standard record script.

When the SO101 keyboard teleop is active, we monkey-patch
``init_keyboard_listener`` to skip the TerminalKeyListener (which conflicts
with pynput) and link recording events to the teleop's arrow-key / ESC handler.

Supports ``--view_mode {mujoco,rerun}`` to select MuJoCo window vs LeRobot
official Rerun visualization. Both modes write the same LeRobot v3.0 dataset.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Register project plugins with LeRobot's ChoiceRegistry before lerobot_record.
# ---------------------------------------------------------------------------
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401
from simstudio.robots.so101_real_follower import SO101RealFollowerConfig  # noqa: F401
from simstudio.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401
from simstudio.teleoperators.so101_keyboard import (  # noqa: F401
    SO101KeyboardTeleopConfig,
)
from simstudio.teleoperators.so101_leader import (  # noqa: F401
    SO101LeaderTeleopConfig,
)

logger = logging.getLogger(__name__)

VIEW_MODES = ("mujoco", "rerun")
DEFAULT_VIEW_MODE = "mujoco"

# Shared recording events for keyboard teleop (linked in connect + init_keyboard_listener).
_recording_events: dict[str, bool] = {
    "exit_early": False,
    "rerecord_episode": False,
    "stop_recording": False,
}


def _detect_teleop_type(argv: list[str] | None = None) -> str | None:
    """Extract teleop.type from the --config YAML file on the command line."""
    args = argv if argv is not None else sys.argv
    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            cfg_path = Path(args[i + 1])
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("teleop", {}).get("type")
    return None


def detect_view_mode(argv: list[str]) -> str:
    """Parse ``--view_mode`` from argv; default ``mujoco``."""
    for i, arg in enumerate(argv):
        if arg == "--view_mode" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--view_mode="):
            return arg.split("=", 1)[1]
    return DEFAULT_VIEW_MODE


def _has_cli_flag(argv: list[str], flag: str) -> bool:
    """Return True if ``flag`` appears as ``--flag`` or ``--flag=value``."""
    prefix = f"{flag}="
    for arg in argv:
        if arg == flag or arg.startswith(prefix):
            return True
    return False


def _strip_cli_flag(argv: list[str], flag: str) -> list[str]:
    """Remove ``--flag`` / ``--flag=value`` and the following value token."""
    stripped: list[str] = []
    skip_next = False
    prefix = f"{flag}="
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg == flag:
            skip_next = True
            continue
        if arg.startswith(prefix):
            continue
        stripped.append(arg)
    return stripped


def apply_view_mode(argv: list[str]) -> list[str]:
    """Map ``--view_mode`` to LeRobot display/render flags and strip ``--view_mode``.

    Explicit ``--display_data``, ``--display_mode``, and ``--robot.render_window``
    flags in *argv* take precedence over view_mode defaults.
    """
    view_mode = detect_view_mode(argv)
    if view_mode not in VIEW_MODES:
        raise ValueError(f"Invalid --view_mode '{view_mode}'. Expected one of {VIEW_MODES}.")

    filtered = _strip_cli_flag(argv, "--view_mode")

    if not _has_cli_flag(filtered, "--display_data"):
        filtered.extend(["--display_data", "true" if view_mode == "rerun" else "false"])

    if view_mode == "rerun" and not _has_cli_flag(filtered, "--display_mode"):
        filtered.extend(["--display_mode", "rerun"])

    if not _has_cli_flag(filtered, "--robot.render_window"):
        filtered.extend(["--robot.render_window", "false" if view_mode == "rerun" else "true"])

    logger.info("view_mode=%s applied to record CLI flags", view_mode)
    return filtered


def _patch_keyboard_recording() -> None:
    """Route keyboard recording controls through SO101 teleop; skip TerminalKeyListener."""
    import lerobot.utils.keyboard_input as _ki

    from simstudio.teleoperators.so101_keyboard import teleop_so101_keyboard as _teleop_module

    def _patched_init_keyboard_listener():
        logging.info(
            "SO101 keyboard teleop active — arrow keys and ESC handle recording "
            "(n/r/q terminal shortcuts disabled to avoid key conflicts)."
        )
        return None, _recording_events

    _ki.init_keyboard_listener = _patched_init_keyboard_listener

    _original_connect = _teleop_module.SO101KeyboardTeleop.connect

    def _patched_connect(self, calibrate: bool = True) -> None:
        _original_connect(self, calibrate)
        self.set_recording_events(_recording_events)

    _teleop_module.SO101KeyboardTeleop.connect = _patched_connect


def _patch_rerun_streaming() -> None:
    """Replace LeRobot ``log_rerun_data`` with a streaming-friendly implementation.

    Upstream logs camera images with ``static=True``, which prevents live updates in
    the Rerun viewer during record loops (camera panes stay black). We keep the same
    blueprint layout and ``init_visualization`` entry point, but stream images on a
    per-frame timeline without ``static=True``.
    """
    import lerobot.utils.rerun_visualization as rv
    import numpy as np
    from lerobot.configs import DEPTH_MILLIMETER_UNIT, infer_depth_unit
    from lerobot.types import RobotAction, RobotObservation
    from lerobot.utils.constants import ACTION, ACTION_PREFIX, OBS_PREFIX, OBS_STR
    from lerobot.utils.import_utils import require_package

    _frame_seq = [0]

    def _streaming_log_rerun_data(
        observation: RobotObservation | None = None,
        action: RobotAction | None = None,
        compress_images: bool = False,
    ) -> None:
        require_package("rerun-sdk", extra="viz", import_name="rerun")
        import rerun as rr

        _frame_seq[0] += 1
        rr.set_time("frame", sequence=_frame_seq[0])

        observation_paths: set[str] = set()
        action_paths: set[str] = set()
        image_paths: set[str] = set()

        if observation:
            for k, v in observation.items():
                if v is None:
                    continue
                key = k if str(k).startswith(OBS_PREFIX) else f"{OBS_STR}.{k}"

                if rv._is_scalar(v):
                    rr.log(key, rr.Scalars(float(v)))
                    observation_paths.add(key)
                elif isinstance(v, np.ndarray):
                    arr = v
                    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
                        arr = np.transpose(arr, (1, 2, 0))
                    if arr.ndim == 1:
                        rr.log(key, rr.Scalars(arr.astype(float)))
                        observation_paths.add(key)
                    elif arr.shape[-1] == 1:
                        depth_unit = infer_depth_unit(arr.dtype)
                        img_entity = rr.DepthImage(
                            arr,
                            meter=1000.0 if depth_unit == DEPTH_MILLIMETER_UNIT else 1.0,
                            colormap=rr.components.Colormap.Viridis,
                        )
                        rr.log(key, entity=img_entity)
                        image_paths.add(key)
                    else:
                        img_entity = rr.Image(arr).compress() if compress_images else rr.Image(arr)
                        rr.log(key, entity=img_entity)
                        image_paths.add(key)

        if action:
            for k, v in action.items():
                if v is None:
                    continue
                key = k if str(k).startswith(ACTION_PREFIX) else f"{ACTION}.{k}"

                if rv._is_scalar(v):
                    rr.log(key, rr.Scalars(float(v)))
                    action_paths.add(key)
                elif isinstance(v, np.ndarray):
                    rr.log(key, rr.Scalars(v.reshape(-1).astype(float)))
                    action_paths.add(key)

        rv._ensure_blueprint(observation_paths, action_paths, image_paths)

    _streaming_log_rerun_data.blueprint = None
    rv.log_rerun_data = _streaming_log_rerun_data
    logging.info("Patched log_rerun_data for streaming Rerun feeds (no static=True)")


if _detect_teleop_type() == "so101_keyboard":
    _patch_keyboard_recording()


def main() -> None:
    import os

    raw_argv = sys.argv[1:]
    view_mode = detect_view_mode(raw_argv)
    argv = apply_view_mode(raw_argv)
    sys.argv = ["lerobot_record.py", *argv]

    if view_mode == "rerun":
        os.environ["SO101_PREFER_EVDEV"] = "1"
        _patch_rerun_streaming()

    # Import AFTER patching so record() sees our patched init_keyboard_listener.
    from lerobot.scripts.lerobot_record import record  # noqa: E402

    record()


if __name__ == "__main__":
    main()
