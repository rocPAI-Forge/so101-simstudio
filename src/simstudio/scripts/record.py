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

import functools
import logging
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _preselect_mujoco_gl(argv: list[str]) -> None:
    """Pick the MuJoCo GL backend BEFORE mujoco is imported (it reads MUJOCO_GL at import).

    - ``--view_mode rerun`` (and any headless/no-window run): use EGL for fast GPU
      offscreen rendering. Software GL caps the record loop around ~15 Hz; EGL sustains 30 Hz.
    - ``--view_mode mujoco``: keep the default GLFW on-screen context. EGL cannot coexist
      with the GLFW window (causes ``GLX X_GLXMakeCurrent BadAccess``), so if the user
      forced ``MUJOCO_GL=egl`` here, override it back to glfw with a warning.
    """
    view_mode = "mujoco"
    for i, arg in enumerate(argv):
        if arg == "--view_mode" and i + 1 < len(argv):
            view_mode = argv[i + 1]
        elif arg.startswith("--view_mode="):
            view_mode = arg.split("=", 1)[1]

    current = os.environ.get("MUJOCO_GL", "")
    if view_mode == "rerun":
        if current == "":
            os.environ["MUJOCO_GL"] = "egl"
    elif current.lower() == "egl":
        logging.getLogger(__name__).warning(
            "MUJOCO_GL=egl conflicts with the GLFW window in --view_mode mujoco; overriding to glfw."
        )
        os.environ["MUJOCO_GL"] = "glfw"


_preselect_mujoco_gl(sys.argv[1:])

# ---------------------------------------------------------------------------
# Register project plugins with LeRobot's ChoiceRegistry before lerobot_record.
# ---------------------------------------------------------------------------
from simstudio.common.constants import KEYBOARD_RECORDING_CONTROLS_HELP  # noqa: E402
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401,E402
from simstudio.robots.so101_real_follower import SO101RealFollowerConfig  # noqa: F401,E402
from simstudio.teleoperators.so101_joycon import SO101JoyConTeleopConfig  # noqa: F401,E402
from simstudio.teleoperators.so101_keyboard import (  # noqa: F401,E402
    SO101KeyboardTeleopConfig,
)
from simstudio.teleoperators.so101_leader import (  # noqa: F401,E402
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

# Tracks multi-episode keyboard record sessions for post-save guards / logging.
_record_session: dict[str, int | None] = {
    "num_episodes": None,
    "saved_this_session": 0,
}

# Kept alive so the evdev recording-control listener (leader/Joy-Con) isn't GC'd.
_evdev_recording_listener = None


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
    """Route all keyboard recording controls through SO101 teleop (single listener)."""
    import lerobot.scripts.lerobot_record as lr
    import lerobot.utils.keyboard_input as _ki

    from simstudio.teleoperators.so101_keyboard import teleop_so101_keyboard as _teleop_module

    if getattr(_patch_keyboard_recording, "_so101_patched", False):
        return

    # Same focus-independent keyboard backend for mujoco and rerun record modes.
    os.environ["SO101_PREFER_EVDEV"] = "1"

    def _patched_init_keyboard_listener():
        logging.info(
            "SO101 keyboard recording controls (%s) handled by teleop only.",
            KEYBOARD_RECORDING_CONTROLS_HELP,
        )
        return None, _recording_events

    _patched_init_keyboard_listener._so101_impl = True
    _ki.init_keyboard_listener = _patched_init_keyboard_listener
    # lerobot_record imports init_keyboard_listener by name; patch that binding too.
    lr.init_keyboard_listener = _patched_init_keyboard_listener

    _original_connect = _teleop_module.SO101KeyboardTeleop.connect

    def _patched_connect(self, calibrate: bool = True) -> None:
        _original_connect(self, calibrate)
        self.set_recording_events(_recording_events)

    _teleop_module.SO101KeyboardTeleop.connect = _patched_connect
    _teleop_module._shared_recording_events = _recording_events
    _patch_keyboard_recording._so101_patched = True


def _install_evdev_recording_controls() -> bool:
    """Drive LeRobot recording controls from a focus-independent evdev listener.

    For non-keyboard teleop (leader arm / Joy-Con) LeRobot falls back to a terminal
    key listener that only registers keys while the terminal window is focused. That
    is impractical when operating a physical leader arm and watching the sim window,
    so the N/Right/R/Left/Q/Esc controls feel unresponsive. Here we reuse the same
    evdev backend as keyboard teleop, which reads ``/dev/input/event*`` regardless of
    window focus, and feed its key presses into the shared recording ``events`` dict.

    Returns True when the evdev listener started (else we leave LeRobot's terminal
    fallback in place).
    """
    global _evdev_recording_listener
    import lerobot.utils.keyboard_input as _ki

    from simstudio.common.recording_controls import apply_keyboard_recording_key
    from simstudio.teleoperators.so101_keyboard.evdev_listener import EvdevKeyListener

    def _on_key(key_name: str, is_pressed: bool) -> None:
        if is_pressed:
            apply_keyboard_recording_key(key_name, _recording_events)

    listener = EvdevKeyListener(_on_key)
    if not listener.start():
        logging.warning(
            "Focus-independent evdev recording controls unavailable (no device or "
            "permission); falling back to terminal input — keep the terminal focused. "
            "For focus-independent keys add your user to the 'input' group: "
            "sudo usermod -aG input $USER"
        )
        return False

    _evdev_recording_listener = listener

    def _patched_init_keyboard_listener():
        logging.info(
            "SO101 recording controls via evdev, focus-independent (%s).",
            KEYBOARD_RECORDING_CONTROLS_HELP,
        )
        return listener, _recording_events

    _patched_init_keyboard_listener._so101_impl = True
    _ki.init_keyboard_listener = _patched_init_keyboard_listener
    _install_evdev_recording_controls._so101_patched = True
    return True


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

    # ``visualization_utils`` binds ``log_rerun_data`` at import time; patching only
    # ``rerun_visualization`` leaves the record loop calling the upstream static=True path.
    import lerobot.utils.visualization_utils as vu

    vu.log_rerun_data = _streaming_log_rerun_data
    logging.info("Patched log_rerun_data for streaming Rerun feeds (no static=True)")


def _maybe_auto_reset_episode(robot: Any, dataset: Any) -> None:
    """Reset MuJoCo arm + block before each recorded episode when configured."""
    if dataset is None or not hasattr(robot, "reset_episode"):
        return

    reset_mode = getattr(getattr(robot, "config", None), "reset_mode", "manual")
    if reset_mode != "auto":
        return

    episode_index = dataset.num_episodes
    logger.info("Auto-resetting sim for episode %s", episode_index)
    robot.reset_episode(episode_index)


# Per-stage timing probe for the record loop (enable with SO101_PROFILE=1).
# Used to locate the FPS bottleneck (rendering vs physics vs dataset write vs serial IO).
_prof_stats: dict[str, list[float]] = {}


def _install_loop_profiling(robot: Any, dataset: Any, teleop: Any) -> None:
    import time as _t

    def _wrap(obj: Any, name: str) -> None:
        if obj is None:
            return
        orig = getattr(obj, name, None)
        if orig is None or getattr(orig, "_so101_prof", False):
            return

        def wrapped(*a: Any, **k: Any):
            t0 = _t.perf_counter()
            try:
                return orig(*a, **k)
            finally:
                s = _prof_stats.setdefault(name, [0.0, 0.0])
                s[0] += 1
                s[1] += _t.perf_counter() - t0

        wrapped._so101_prof = True  # type: ignore[attr-defined]
        setattr(obj, name, wrapped)

    _wrap(robot, "get_observation")
    _wrap(robot, "send_action")
    _wrap(dataset, "add_frame")
    _wrap(teleop, "get_action")


def _log_prof_stats() -> None:
    if not _prof_stats:
        return
    parts = []
    for label, (n, tot) in sorted(_prof_stats.items(), key=lambda x: -x[1][1]):
        if n:
            parts.append(f"{label}={tot / n * 1e3:.1f}ms x{int(n)}")
    logger.info("SO101_PROFILE per-frame avg: %s", " | ".join(parts))


def _patch_episode_auto_reset() -> None:
    """Hook LeRobot record_loop to auto-reset MuJoCo sim before each episode."""
    import lerobot.scripts.lerobot_record as lr

    from simstudio.teleoperators.so101_keyboard.teleop_so101_keyboard import (
        set_recording_keys_enabled,
    )

    if getattr(lr.record_loop, "_so101_episode_reset_patched", False):
        return

    _original_record_loop = lr.record_loop

    @functools.wraps(_original_record_loop)
    def _patched_record_loop(*args: Any, **kwargs: Any):
        robot = kwargs.get("robot")
        if robot is None and args:
            robot = args[0]
        dataset = kwargs.get("dataset")
        teleop = kwargs.get("teleop")
        _maybe_auto_reset_episode(robot, dataset)
        _profile = os.environ.get("SO101_PROFILE") == "1"
        if _profile:
            _install_loop_profiling(robot, dataset, teleop)
            _prof_stats.clear()
        # Enable recording hotkeys only while actively looping; keeping them off during
        # the blocking video encode in save_episode prevents a stray key from setting
        # stop_recording between episodes.
        set_recording_keys_enabled(True)
        try:
            return _original_record_loop(*args, **kwargs)
        finally:
            set_recording_keys_enabled(False)
            if _profile:
                _log_prof_stats()

    _patched_record_loop._so101_episode_reset_patched = True
    lr.record_loop = _patched_record_loop


def _init_record_session_from_argv(argv: list[str]) -> None:
    """Seed session counters from CLI flags or config YAML."""
    num_episodes: int | None = None
    for i, arg in enumerate(argv):
        if arg == "--dataset.num_episodes" and i + 1 < len(argv):
            num_episodes = int(argv[i + 1])
            break
        if arg.startswith("--dataset.num_episodes="):
            num_episodes = int(arg.split("=", 1)[1])
            break

    if num_episodes is None:
        for i, arg in enumerate(argv):
            if arg == "--config" and i + 1 < len(argv):
                cfg_path = Path(argv[i + 1])
                if cfg_path.exists():
                    with open(cfg_path) as f:
                        cfg = yaml.safe_load(f)
                    num_episodes = cfg.get("dataset", {}).get("num_episodes")
                break

    _record_session["num_episodes"] = num_episodes
    _record_session["saved_this_session"] = 0
    if num_episodes is not None:
        logger.info(
            "Record session: %s episode(s) this run (→/N save & next, ←/R re-record, ESC/Q stop all)",
            num_episodes,
        )


def _patch_skip_empty_episode_save() -> None:
    """Skip save_episode when the buffer is empty (e.g. ESC before any frames)."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if getattr(LeRobotDataset.save_episode, "_so101_skip_empty_patched", False):
        return

    _original_save_episode = LeRobotDataset.save_episode

    @functools.wraps(_original_save_episode)
    def _patched_save_episode(
        self,
        episode_data: Any = None,
        parallel_encoding: bool = True,
    ) -> None:
        if episode_data is None and not self.has_pending_frames():
            logger.info(
                "Skipping save_episode: episode buffer is empty "
                "(early stop or cancelled before any frames were recorded)"
            )
            self.clear_episode_buffer()
            return
        _original_save_episode(self, episode_data, parallel_encoding)
        _record_session["saved_this_session"] = int(_record_session.get("saved_this_session", 0)) + 1
        logger.info(
            "Episode saved (%s/%s this run); stop_recording=%s",
            _record_session.get("saved_this_session"),
            _record_session.get("num_episodes"),
            _recording_events.get("stop_recording"),
        )

    _patched_save_episode._so101_skip_empty_patched = True
    LeRobotDataset.save_episode = _patched_save_episode


_patch_episode_auto_reset()
_patch_skip_empty_episode_save()

if _detect_teleop_type() == "so101_keyboard":
    _patch_keyboard_recording()


def main() -> None:
    raw_argv = sys.argv[1:]
    view_mode = detect_view_mode(raw_argv)
    argv = apply_view_mode(raw_argv)
    sys.argv = ["lerobot_record.py", *argv]

    if _detect_teleop_type(raw_argv) == "so101_keyboard":
        _patch_keyboard_recording()
    else:
        # leader arm / Joy-Con: use focus-independent evdev controls instead of the
        # terminal fallback so N/Right/R/Left/Q/Esc work without terminal focus.
        _install_evdev_recording_controls()

    _init_record_session_from_argv(raw_argv)
    _recording_events["stop_recording"] = False
    _recording_events["exit_early"] = False
    _recording_events["rerecord_episode"] = False

    if view_mode == "rerun":
        _patch_rerun_streaming()

    # Import AFTER patching so record() sees our patched init_keyboard_listener.
    import lerobot.scripts.lerobot_record as lr
    import lerobot.utils.keyboard_input as ki
    from lerobot.scripts.lerobot_record import record  # noqa: E402

    if getattr(_patch_keyboard_recording, "_so101_patched", False) or getattr(
        _install_evdev_recording_controls, "_so101_patched", False
    ):
        lr.init_keyboard_listener = ki.init_keyboard_listener

    record()


if __name__ == "__main__":
    main()
