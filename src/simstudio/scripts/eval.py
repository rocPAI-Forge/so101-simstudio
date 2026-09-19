"""Sim2sim closed-loop policy evaluation in MuJoCo.

Runs a trained policy for multiple episodes with random cube resets,
reports per-episode success (cube in container) and aggregate success rate.

Uses LeRobot's rollout context (policy + pre/post processors + inference
engine) but owns the multi-episode loop so no ``rollout_`` dataset is required.
"""

import logging
import os
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml


def _preselect_mujoco_gl_eval(argv: list[str]) -> None:
    """Use EGL offscreen rendering when GLFW window is disabled or DISPLAY is missing."""
    headless = not os.environ.get("DISPLAY")
    for i, arg in enumerate(argv):
        if arg == "--robot.render_window" and i + 1 < len(argv):
            headless = headless or argv[i + 1].lower() in ("false", "0")
        elif arg.startswith("--robot.render_window="):
            headless = headless or arg.split("=", 1)[1].lower() in ("false", "0")
    if headless and not os.environ.get("MUJOCO_GL"):
        os.environ["MUJOCO_GL"] = "egl"


_preselect_mujoco_gl_eval(sys.argv[1:])

# Rollout imports DAgger → pynput → python-xlib. Display() can block forever on
# GNOME Xwayland even when GLFW/xset work. Hide DISPLAY only during import.
_eval_display = os.environ.pop("DISPLAY", None)
try:
    from lerobot.configs import parser  # noqa: E402
    from lerobot.rollout import BaseStrategyConfig, RolloutConfig, build_rollout_context  # noqa: E402
    from lerobot.rollout.strategies.base import BaseStrategy  # noqa: E402
    from lerobot.rollout.strategies.core import send_next_action  # noqa: E402
    from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402
    from lerobot.utils.process import ProcessSignalHandler  # noqa: E402
    from lerobot.utils.robot_utils import precise_sleep  # noqa: E402
    from lerobot.utils.utils import init_logging, log_say  # noqa: E402
    from lerobot.utils.visualization_utils import init_visualization, shutdown_visualization  # noqa: E402
finally:
    if _eval_display is not None:
        os.environ["DISPLAY"] = _eval_display

from simstudio.common.eval_success import (  # noqa: E402
    check_pick_success,
    cube_over_container,
    gripper_near_cube,
)
from simstudio.robots.so101_mujoco import SO101MujocoConfig  # noqa: F401,E402
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_EVAL = {
    "num_episodes": 10,
    "episode_time_s": 60.0,
    "reset_time_s": 3.0,
    "stats_dataset_repo_id": "alexhegit/so101-simstudio-lab01-pnp",
    "stats_dataset_root": "./datasets/so101-simstudio-lab01-pnp",
    "state_dim": 6,
}

_EVAL_SETTINGS: dict[str, Any] = {}


def _config_path_from_argv(argv: list[str]) -> Path:
    for i, arg in enumerate(argv):
        if arg == "--config" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--config="):
            return Path(arg.split("=", 1)[1])
    print("Usage: python -m simstudio.scripts.eval --config <yaml> --policy.path <checkpoint>")
    sys.exit(1)


def _parse_eval_overrides(argv: list[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if not arg.startswith("--eval."):
            i += 1
            continue
        field = arg.removeprefix("--eval.")
        if "=" in field:
            name, value = field.split("=", 1)
        elif i + 1 < len(argv):
            name = field
            value = argv[i + 1]
            i += 1
        else:
            i += 1
            continue
        if name == "num_episodes":
            overrides[name] = int(value)
        elif name in ("episode_time_s", "reset_time_s"):
            overrides[name] = float(value)
        elif name in ("stats_dataset_repo_id", "stats_dataset_root"):
            overrides[name] = value
        i += 1
    return overrides


def _prepare_argv_for_rollout_parser() -> None:
    """Map ``--config`` to LeRobot ``--config_path`` and drop the ``eval:`` yaml block."""
    argv = sys.argv[1:]
    config_path = _config_path_from_argv(argv)
    eval_overrides = _parse_eval_overrides(argv)

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    global _EVAL_SETTINGS
    _EVAL_SETTINGS = {**DEFAULT_EVAL, **raw.pop("eval", {}), **eval_overrides}
    rollout_doc = raw
    rollout_doc.setdefault("strategy", {"type": "base"})

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(rollout_doc, tmp, sort_keys=False)
    tmp.close()

    new_argv = [sys.argv[0], f"--config_path={tmp.name}"]
    skip_next = False
    for arg in argv:
        if skip_next:
            skip_next = False
            continue
        if arg in ("--config",):
            skip_next = True
            continue
        if arg.startswith("--config="):
            continue
        if arg.startswith("--eval."):
            if "=" not in arg:
                skip_next = True
            continue
        new_argv.append(arg)
    sys.argv = new_argv


class _EpisodeStageTracker:
    """Track where a pick-and-place rollout stops making progress.

    Distances use the fingertip ``gripperframe`` site. Thresholds are diagnostic
    and deliberately looser than final success: reach <=8 cm, near <=4 cm,
    lift >=2 cm above the episode's initial cube height.
    """

    def __init__(self, robot: SO101MujocoRobot) -> None:
        import mujoco as mj

        self.robot = robot
        site_name = "gripperframe"
        self.site_id = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            self.site_id = robot.ee_site_id
        block = robot.get_block_position()
        self.initial_z = block[2] if block is not None else 0.015
        self.min_gripper_cube_m = float("inf")
        self.max_cube_z_m = self.initial_z
        self.min_container_xy_m = float("inf")
        self.seen_open = False
        self.stages = {
            "reach_8cm": False,
            "near_4cm": False,
            "close_after_open": False,
            "lift_2cm": False,
            "over_box": False,
            "released_after_lift": False,
            "placed_at_any_tick": False,
            "dropped_after_lift": False,
        }

    def update(self, sent: dict[str, Any] | None) -> None:
        from simstudio.common.eval_success import CONTAINER_XY

        block = self.robot.get_block_position()
        if block is None:
            return
        site = self.robot.data.site_xpos[self.site_id]
        dist = (
            (float(site[0]) - block[0]) ** 2
            + (float(site[1]) - block[1]) ** 2
            + (float(site[2]) - block[2]) ** 2
        ) ** 0.5
        self.min_gripper_cube_m = min(self.min_gripper_cube_m, dist)
        self.max_cube_z_m = max(self.max_cube_z_m, block[2])
        box_xy = ((block[0] - CONTAINER_XY[0]) ** 2 + (block[1] - CONTAINER_XY[1]) ** 2) ** 0.5
        self.min_container_xy_m = min(self.min_container_xy_m, box_xy)
        self.stages["reach_8cm"] |= dist <= 0.08
        self.stages["near_4cm"] |= dist <= 0.04
        self.stages["lift_2cm"] |= (
            self.stages["close_after_open"] and block[2] >= self.initial_z + 0.02
        )
        self.stages["over_box"] |= self.stages["lift_2cm"] and cube_over_container(block)
        self.stages["placed_at_any_tick"] |= (
            self.stages["released_after_lift"] and check_pick_success(block)
        )
        if (
            self.stages["lift_2cm"]
            and block[2] <= self.initial_z + 0.005
            and not cube_over_container(block)
        ):
            self.stages["dropped_after_lift"] = True

        if sent is None:
            return
        gripper = sent.get("gripper.pos")
        if gripper is None:
            return
        gripper = float(gripper)
        if gripper >= 0.45:
            if self.stages["lift_2cm"] and self.stages["over_box"]:
                self.stages["released_after_lift"] = True
            self.seen_open = True
        elif self.seen_open and self.stages["near_4cm"] and gripper <= 0.15:
            self.stages["close_after_open"] = True

    def result(self) -> dict[str, Any]:
        return {
            **self.stages,
            "min_gripper_cube_m": self.min_gripper_cube_m,
            "max_cube_z_m": self.max_cube_z_m,
            "min_container_xy_m": self.min_container_xy_m,
        }


def _run_episode(
    strategy: BaseStrategy,
    ctx: Any,
    episode_time_s: float,
) -> dict[str, Any]:
    cfg = ctx.runtime.cfg
    robot = ctx.hardware.robot_wrapper
    engine = strategy._engine
    interpolator = strategy._interpolator
    control_interval = interpolator.get_control_interval(cfg.fps)
    inner = getattr(robot, "inner", robot)
    stage_tracker = _EpisodeStageTracker(inner)
    try:
        debug_n = int(os.environ.get("LAB01_DEBUG_ACTIONS", "0") or "0")
    except ValueError:
        debug_n = 0
    debug_left = [debug_n]

    infer_n = 0
    infer_s = 0.0
    chunk_n = 0
    chunk_s = 0.0
    tick_n = 0
    tick_s = 0.0
    orig_get_action = engine.get_action
    policy = getattr(getattr(ctx, "policy", None), "policy", None)
    orig_predict = getattr(policy, "predict_action_chunk", None) if policy is not None else None

    def timed_get_action(obs_frame: dict | None):
        nonlocal infer_n, infer_s
        t0 = time.perf_counter()
        out = orig_get_action(obs_frame)
        infer_s += time.perf_counter() - t0
        infer_n += 1
        return out

    def timed_predict_chunk(*args, **kwargs):
        nonlocal chunk_n, chunk_s
        t0 = time.perf_counter()
        out = orig_predict(*args, **kwargs)
        chunk_s += time.perf_counter() - t0
        chunk_n += 1
        return out

    engine.get_action = timed_get_action  # type: ignore[method-assign]
    if orig_predict is not None:
        policy.predict_action_chunk = timed_predict_chunk  # type: ignore[method-assign]
    start = time.perf_counter()
    engine.resume()
    try:
        while not ctx.runtime.shutdown_event.is_set():
            if time.perf_counter() - start >= episode_time_s:
                break

            loop_start = time.perf_counter()
            obs = robot.get_observation()
            obs_processed = strategy._process_observation_and_notify(ctx.processors, obs)

            if strategy._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            sent = send_next_action(obs_processed, obs, ctx, interpolator)
            stage_tracker.update(sent)
            if debug_left[0] > 0 and sent is not None:
                debug_left[0] -= 1
                names = list(getattr(inner, "JOINT_NAMES", []))
                current = [float(obs.get(f"{n}.pos", float("nan"))) for n in names]
                pred = [float(sent.get(f"{n}.pos", sent.get(n, float("nan")))) for n in names]
                delta = [p - q for p, q in zip(pred, current, strict=False)]
                logger.info(
                    "debug action[%d] pred=%s current=%s delta=%s",
                    debug_n - debug_left[0],
                    [round(v, 4) for v in pred],
                    [round(v, 4) for v in current],
                    [round(v, 4) for v in delta],
                )

            dt = time.perf_counter() - loop_start
            tick_n += 1
            tick_s += dt
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
    finally:
        engine.get_action = orig_get_action  # type: ignore[method-assign]
        if orig_predict is not None:
            policy.predict_action_chunk = orig_predict  # type: ignore[method-assign]

    return {
        "infer_n": float(infer_n),
        "infer_s": infer_s,
        "chunk_n": float(chunk_n),
        "chunk_s": chunk_s,
        "tick_n": float(tick_n),
        "tick_s": tick_s,
        "funnel": stage_tracker.result(),
    }


def _reset_pause(robot: SO101MujocoRobot, reset_time_s: float, fps: float) -> None:
    steps = max(int(reset_time_s * fps), 1)
    for _ in range(steps):
        if robot.config.render_window:
            robot._render_glfw()
        precise_sleep(1.0 / fps)


def _policy_state_dim(ctx: Any) -> int | None:
    """Return the policy's expected ``observation.state`` dimension, if any."""
    policy_cfg = ctx.policy.policy.config
    robot_state = getattr(policy_cfg, "robot_state_feature", None)
    if robot_state is not None:
        return int(robot_state.shape[0])
    input_features = getattr(policy_cfg, "input_features", {}) or {}
    state_feat = input_features.get("observation.state")
    if state_feat is not None:
        return int(state_feat.shape[0])
    return None


def _molmoact2_normalizer_step(ctx: Any) -> Any | None:
    """Return the MolmoAct2 masked normalizer step, if the policy uses one."""
    for step in ctx.policy.preprocessor.steps:
        if type(step).__name__.startswith("MolmoAct2Masked"):
            return step
    return None


def _fitted_state_dim(step: Any) -> int | None:
    """Return the ``observation.state`` dim the normalizer was fitted on."""
    stats = getattr(step, "stats", None)
    if not isinstance(stats, dict):
        return None
    feature_stats = stats.get("observation.state")
    if not isinstance(feature_stats, dict):
        return None
    for key in ("mean", "q01", "min"):
        value = feature_stats.get(key)
        if value is None:
            continue
        shape = getattr(value, "shape", None)
        return int(shape[-1]) if shape else len(value)
    return None


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes")


def _install_gripper_snap(ctx: Any) -> None:
    """Optionally snap the predicted gripper command to open/closed at eval time.

    Lab 01 grasps only hold when the jaws reach the safe closed value (~-0.1 rad);
    intermediate commands look like the arm hesitating over the cube. Flow-matching
    heads (VLA-JEPA) regress the leader's continuous open->close sweep and can settle
    mid-range. This is a diagnostic knob, not a training fix: quote it next to any
    success rate measured with it on.

    LIBERO's own ``binarize_gripper_action`` cannot be reused — it targets dim 6 of a
    7-D action and maps to {-1, +1}, while SO-101 is dim 5 of 6 in radians.
    """
    import torch
    from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey

    threshold = float(os.environ.get("LAB01_GRIPPER_SNAP_THRESHOLD", "0.45"))
    closed = float(os.environ.get("LAB01_GRIPPER_SNAP_CLOSED", "-0.1"))
    open_raw = os.environ.get("LAB01_GRIPPER_SNAP_OPEN", "").strip()
    open_value = float(open_raw) if open_raw else None
    latch = _env_flag("LAB01_GRIPPER_SNAP_LATCH")

    class _SO101GripperSnapStep(ProcessorStep):
        def __init__(self) -> None:
            self._seen_open = False
            self._latched = False

        def __call__(self, transition: EnvTransition) -> EnvTransition:
            action = transition.get(TransitionKey.ACTION)
            if action is None or not isinstance(action, torch.Tensor):
                return transition
            if action.shape[-1] < 6:
                return transition
            a = action.clone()
            wants_close = bool((a[..., -1] < threshold).all())
            if not wants_close:
                self._seen_open = True
            # Episodes start with the jaws near closed, so latching before the
            # policy has opened once would freeze the arm shut for the whole run.
            if wants_close and self._seen_open and latch:
                self._latched = True
            if wants_close or self._latched:
                a[..., -1] = closed
            elif open_value is not None:
                a[..., -1] = open_value
            transition = dict(transition)
            transition[TransitionKey.ACTION] = a
            return transition

        def reset(self) -> None:
            self._seen_open = False
            self._latched = False

        def transform_features(self, features):
            return features

    ctx.policy.postprocessor.steps.append(_SO101GripperSnapStep())
    logger.info(
        "Gripper snap enabled: threshold=%.3f closed=%.3f open=%s latch=%s",
        threshold,
        closed,
        open_value if open_value is not None else "<predicted>",
        latch,
    )


class _GripperProximityOracle:
    """Force-close when the wrist is over the cube; open over the container.

    Diagnostic only: proves whether the policy's arm trajectory is enough if the
    gripper channel is replaced by geometry. Quote it next to any success rate.
    """

    def __init__(self, robot: SO101MujocoRobot) -> None:
        self.robot = robot
        self.radius_m = float(os.environ.get("LAB01_GRIPPER_ORACLE_RADIUS", "0.04"))
        self.closed_val = float(os.environ.get("LAB01_GRIPPER_ORACLE_CLOSED", "-0.1"))
        self.open_val = float(os.environ.get("LAB01_GRIPPER_ORACLE_OPEN", "0.9"))
        site = os.environ.get("LAB01_GRIPPER_ORACLE_SITE", "gripperframe")
        import mujoco as mj

        self.site_id = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_SITE, site)
        if self.site_id < 0:
            logger.warning("Gripper oracle site %r missing; falling back to wrist_site", site)
            self.site_id = robot.ee_site_id
        self.reset()

    def reset(self) -> None:
        self.holding = False
        self.released = False
        self.min_dist = float("inf")

    def _site_xyz(self) -> tuple[float, float, float]:
        p = self.robot.data.site_xpos[self.site_id]
        return (float(p[0]), float(p[1]), float(p[2]))

    def override(self, action: dict[str, Any], robot: SO101MujocoRobot) -> dict[str, Any]:
        block = robot.get_block_position()
        if check_pick_success(block):
            return action
        ee_xyz = self._site_xyz()
        if block is not None:
            dx = ee_xyz[0] - block[0]
            dy = ee_xyz[1] - block[1]
            dz = ee_xyz[2] - block[2]
            dist = (dx * dx + dy * dy + dz * dz) ** 0.5
            if dist < self.min_dist:
                self.min_dist = dist
        near = gripper_near_cube(ee_xyz, block, radius_m=self.radius_m)
        over_box = cube_over_container(block)
        if self.holding and over_box:
            self.holding = False
            self.released = True
            logger.info("Gripper oracle: open over container (ee=%s cube=%s)", ee_xyz, block)
        elif near and not self.holding and not self.released:
            self.holding = True
            logger.info(
                "Gripper oracle: close (dist<=%.3f ee=%s cube=%s)",
                self.radius_m,
                ee_xyz,
                block,
            )
        out = dict(action)
        if self.holding:
            out["gripper.pos"] = self.closed_val
        elif self.released:
            out["gripper.pos"] = self.open_val
        return out


def _install_gripper_oracle(robot: SO101MujocoRobot) -> _GripperProximityOracle:
    oracle = _GripperProximityOracle(robot)
    orig = robot.send_action

    def wrapped(action: dict[str, Any]) -> Any:
        return orig(oracle.override(action, robot))

    robot.send_action = wrapped  # type: ignore[method-assign]
    logger.info(
        "Gripper oracle enabled: radius=%.3f close=%.3f open=%.3f site_id=%d",
        oracle.radius_m,
        oracle.closed_val,
        oracle.open_val,
        oracle.site_id,
    )
    return oracle


def _preserve_vla_jepa_finetune_dims() -> None:
    """Keep Lab01 VLA-JEPA ``state_dim`` when LIBERO ``input_features`` still say 8.

    Fine-tunes store ``state_dim`` 6 or 15 (and a matching state encoder) but
    leave the base LIBERO ``observation.state`` shape at 8. ``validate_features``
    would clobber that, build an 8-d encoder, then skip the fine-tune weights
    (``strict=False``). Restore whenever the saved dim differs — not only when
    it is larger than 8 (6-D Lab 01 slice is smaller).
    """
    from lerobot.policies.vla_jepa.configuration_vla_jepa import VLAJEPAConfig

    orig = VLAJEPAConfig.validate_features

    def wrapped(self) -> None:
        saved_state = int(self.state_dim)
        saved_action = int(self.action_dim)
        orig(self)
        if saved_state != self.state_dim:
            logger.info(
                "Keeping VLA-JEPA state_dim=%d (input_features had %s)",
                saved_state,
                self.state_dim,
            )
            self.state_dim = saved_state
            feat = (self.input_features or {}).get("observation.state")
            if feat is not None:
                self.input_features["observation.state"] = replace(feat, shape=(saved_state,))
        if saved_action and saved_action != self.action_dim:
            self.action_dim = saved_action

    VLAJEPAConfig.validate_features = wrapped  # type: ignore[method-assign]


def _checkpoint_normalizer_fitted_state_dim(ctx: Any) -> int | None:
    """Return ``observation.state`` dim from the loaded checkpoint normalizer."""
    molmo_step = _molmoact2_normalizer_step(ctx)
    if molmo_step is not None:
        return _fitted_state_dim(molmo_step)
    from lerobot.processor.normalize_processor import NormalizerProcessorStep

    for step in ctx.policy.preprocessor.steps:
        if type(step) is NormalizerProcessorStep:
            return _fitted_state_dim(step)
    return None


def _fix_groot_inference_timesteps(ctx: Any) -> None:
    """LeRobot GrootConfig stores ``num_inference_timesteps: null``.

    Loading a Lab 01 continuation checkpoint as ``base_model_path`` can copy
    that null onto the inner N1.7 model, which then crashes in
    ``dt = 1.0 / self.num_inference_timesteps``. NVIDIA's N1.7 default is 4.
    """
    policy = getattr(getattr(ctx, "policy", None), "policy", None)
    if policy is None:
        return
    groot = getattr(policy, "_groot_model", None)
    if groot is None:
        return
    head = getattr(groot, "action_head", None)
    patched = []
    for obj in (head, groot, getattr(head, "config", None) if head else None, getattr(groot, "config", None)):
        if obj is None:
            continue
        n = getattr(obj, "num_inference_timesteps", "missing")
        if n in (None, 0, "missing") and (hasattr(obj, "num_inference_timesteps") or obj is head):
            obj.num_inference_timesteps = 4
            patched.append(type(obj).__name__)
        horizon = getattr(obj, "action_horizon", "missing")
        if horizon in (None, 0) and hasattr(obj, "action_horizon"):
            obj.action_horizon = 16
            patched.append(f"{type(obj).__name__}.action_horizon")
    if patched:
        logger.info("Patched GR00T inference knobs on %s", patched)


def _expand_rollout_state_features(ctx: Any, robot: Any) -> None:
    """Include vel/ee in rollout ``observation.state`` when the policy expects them.

    LeRobot rollout only routes ``*.pos`` joint keys to the policy by default.
    ACT trained on our dataset uses 15-dim state (pos+vel+ee).

    MolmoAct2 / VLA-JEPA fine-tunes may keep the base checkpoint's smaller
    ``input_features`` even though training fed the full dataset state, so
    trust the dim the checkpoint normalizer was actually fitted on.
    """
    from lerobot.utils.constants import OBS_STR
    from lerobot.utils.feature_utils import hw_to_dataset_features

    policy_dim = _policy_state_dim(ctx)
    fitted_dim = _checkpoint_normalizer_fitted_state_dim(ctx)
    if fitted_dim is not None and fitted_dim != policy_dim:
        logger.info(
            "Checkpoint normalizer was fitted on %d state dims "
            "(policy config says %s); using %d",
            fitted_dim,
            policy_dim,
            fitted_dim,
        )
        policy_dim = fitted_dim
    if policy_dim is None:
        return

    current = ctx.data.dataset_features.get("observation.state", {})
    current_dim = int(current.get("shape", [0])[0]) if current else 0
    if current_dim >= policy_dim:
        return

    obs_hw = {
        k: v
        for k, v in robot.observation_features.items()
        if isinstance(v, tuple) or v is float
    }
    state_features = hw_to_dataset_features(obs_hw, OBS_STR)
    state_names = state_features["observation.state"]["names"]
    if len(state_names) < policy_dim:
        logger.warning(
            "Robot observation has %d state keys but policy expects %d",
            len(state_names),
            policy_dim,
        )
        return

    ctx.data.dataset_features.update(state_features)
    ctx.data.hw_features.update(state_features)
    logger.info(
        "Expanded rollout observation.state from %d to %d dims for policy inference",
        current_dim,
        policy_dim,
    )


def _fix_preprocessor_normalizer_stats(ctx: Any, eval_settings: dict[str, Any], rename_map: dict) -> None:
    """Align normalizer ``observation.state`` stats with the policy input dimension.

    SmolVLA uses 6 joint positions at inference; slice 15-dim dataset stats.
    ACT uses the full 15-dim dataset state — keep stats unsliced.
    """
    from lerobot.datasets import LeRobotDatasetMetadata
    from lerobot.processor.normalize_processor import NormalizerProcessorStep
    from lerobot.processor.rename_processor import rename_stats

    repo_id = eval_settings["stats_dataset_repo_id"]
    root = eval_settings.get("stats_dataset_root")
    state_dim = _policy_state_dim(ctx) or int(eval_settings.get("state_dim", 6))
    fitted_dim = _checkpoint_normalizer_fitted_state_dim(ctx)
    if fitted_dim is not None and fitted_dim != state_dim:
        logger.info(
            "Using checkpoint normalizer state dim %d instead of policy config %d",
            fitted_dim,
            state_dim,
        )
        state_dim = fitted_dim

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root)
    stats = rename_stats(ds_meta.stats, rename_map)
    if "observation.state" in stats:
        stats["observation.state"] = {
            k: v[:state_dim] for k, v in stats["observation.state"].items()
        }

    for step in ctx.policy.preprocessor.steps:
        # SmolVLA uses NormalizerProcessorStep. MolmoAct2 subclasses it
        # (molmoact2_masked_normalizer); overwriting that step drops the
        # checkpoint quantile stats / mask and can collapse closed-loop actions.
        if type(step) is not NormalizerProcessorStep:
            continue
        step.stats = stats
        step.to(device=step.device)
        logger.info(
            "Patched normalizer stats from %s (observation.state dim=%d)",
            repo_id,
            state_dim,
        )
        return
    logger.warning("NormalizerProcessorStep not found; skipping stats patch")


@parser.wrap()
def eval_main(cfg: RolloutConfig) -> None:
    if not isinstance(cfg.strategy, BaseStrategyConfig):
        raise ValueError("eval.py requires strategy.type=base (multi-episode loop is built-in)")

    eval_settings = _EVAL_SETTINGS
    num_episodes = int(eval_settings["num_episodes"])
    episode_time_s = float(eval_settings["episode_time_s"])
    reset_time_s = float(eval_settings["reset_time_s"])

    if cfg.display_data:
        init_visualization(
            cfg.display_mode,
            session_name="eval",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )

    signal_handler = ProcessSignalHandler(use_threads=True, display_pid=False)
    _preserve_vla_jepa_finetune_dims()
    ctx = build_rollout_context(cfg, signal_handler.shutdown_event)
    _fix_groot_inference_timesteps(ctx)
    robot = ctx.hardware.robot_wrapper
    inner_robot = getattr(robot, "inner", robot)
    _expand_rollout_state_features(ctx, inner_robot)
    _fix_preprocessor_normalizer_stats(ctx, eval_settings, cfg.rename_map)
    if _env_flag("LAB01_GRIPPER_SNAP"):
        _install_gripper_snap(ctx)
    inner = getattr(robot, "inner", robot)
    if not isinstance(inner, SO101MujocoRobot):
        raise TypeError(f"eval.py expects so101_mujoco robot, got {type(inner)}")
    gripper_oracle = _install_gripper_oracle(inner) if _env_flag("LAB01_GRIPPER_ORACLE") else None

    strategy = BaseStrategy(cfg.strategy)
    results: list[dict[str, Any]] = []

    logger.info(
        "Starting sim2sim eval: %d episodes × %.0fs @ %.0f Hz | policy=%s",
        num_episodes,
        episode_time_s,
        cfg.fps,
        cfg.policy.pretrained_path,
    )
    log_say(f"Sim2sim eval: {num_episodes} episodes", cfg.play_sounds)

    try:
        strategy.setup(ctx)
        for ep in range(num_episodes):
            if ctx.runtime.shutdown_event.is_set():
                break

            log_say(f"Eval episode {ep + 1} of {num_episodes}", cfg.play_sounds)
            if getattr(inner.config, "reset_mode", "manual") == "auto":
                inner.reset_episode(ep)

            if gripper_oracle is not None:
                gripper_oracle.reset()
            strategy._engine.reset()
            strategy._interpolator.reset()
            timing = _run_episode(strategy, ctx, episode_time_s)

            if gripper_oracle is not None:
                logger.info(
                    "Gripper oracle ep %d: holding=%s released=%s min_dist=%.3f",
                    ep,
                    gripper_oracle.holding,
                    gripper_oracle.released,
                    gripper_oracle.min_dist,
                )

            block_pos = inner.get_block_position()
            success = check_pick_success(block_pos)
            infer_hz = (timing["infer_n"] / timing["infer_s"]) if timing["infer_s"] > 0 else 0.0
            chunk_hz = (timing["chunk_n"] / timing["chunk_s"]) if timing.get("chunk_s", 0.0) > 0 else 0.0
            funnel = timing.pop("funnel")
            funnel["final_success"] = success
            results.append(
                {
                    "episode": ep,
                    "success": success,
                    "block_pos": block_pos,
                    "funnel": funnel,
                    **timing,
                }
            )
            status = "SUCCESS" if success else "fail"
            pos_str = f"{block_pos}" if block_pos else "unknown"
            logger.info(
                "Episode %d: %s (block=%s) policy_infer=%.2f Hz (%d calls, %.1fs) "
                "chunk=%.2f Hz (%d calls, %.1fs, mean %.0fms)",
                ep,
                status,
                pos_str,
                infer_hz,
                int(timing["infer_n"]),
                timing["infer_s"],
                chunk_hz,
                int(timing.get("chunk_n", 0.0)),
                float(timing.get("chunk_s", 0.0)),
                (1000.0 * timing["chunk_s"] / timing["chunk_n"]) if timing.get("chunk_n", 0.0) else 0.0,
            )
            logger.info(
                "Episode %d funnel: %s",
                ep,
                yaml.safe_dump(funnel, default_flow_style=True).strip(),
            )
            print(
                f"Episode {ep + 1}/{num_episodes}: {status}  block={pos_str}  "
                f"infer={infer_hz:.2f} Hz  chunk={chunk_hz:.2f} Hz"
            )

            if ep < num_episodes - 1 and reset_time_s > 0:
                _reset_pause(inner, reset_time_s, cfg.fps)
    finally:
        strategy.teardown(ctx)
        if cfg.display_data:
            shutdown_visualization()

    if results:
        n_success = sum(1 for r in results if r["success"])
        rate = 100.0 * n_success / len(results)
        infer_n = sum(float(r.get("infer_n", 0.0)) for r in results)
        infer_s = sum(float(r.get("infer_s", 0.0)) for r in results)
        chunk_n = sum(float(r.get("chunk_n", 0.0)) for r in results)
        chunk_s = sum(float(r.get("chunk_s", 0.0)) for r in results)
        tick_n = sum(float(r.get("tick_n", 0.0)) for r in results)
        tick_s = sum(float(r.get("tick_s", 0.0)) for r in results)
        infer_hz = (infer_n / infer_s) if infer_s > 0 else 0.0
        chunk_hz = (chunk_n / chunk_s) if chunk_s > 0 else 0.0
        tick_hz = (tick_n / tick_s) if tick_s > 0 else 0.0
        chunk_ms = (1000.0 * chunk_s / chunk_n) if chunk_n else 0.0
        summary = (
            f"Eval complete: {n_success}/{len(results)} success ({rate:.0f}%) | "
            f"policy infer {infer_hz:.2f} Hz ({int(infer_n)} calls, {infer_s:.1f}s) | "
            f"chunk {chunk_hz:.2f} Hz ({int(chunk_n)} calls, mean {chunk_ms:.0f}ms) | "
            f"control compute {tick_hz:.1f} Hz (excl. fps sleep)"
        )
        logger.info(summary)
        print(summary)
        funnel_keys = [
            "reach_8cm",
            "near_4cm",
            "close_after_open",
            "lift_2cm",
            "over_box",
            "released_after_lift",
            "placed_at_any_tick",
            "final_success",
            "dropped_after_lift",
        ]
        funnel_counts = {
            key: sum(bool(result["funnel"][key]) for result in results) for key in funnel_keys
        }
        funnel_summary = (
            f"Stage funnel ({len(results)} ep): "
            + " → ".join(
                f"{key}={funnel_counts[key]}/{len(results)}"
                for key in funnel_keys
                if key != "dropped_after_lift"
            )
            + f" | dropped_after_lift={funnel_counts['dropped_after_lift']}/{len(results)}"
        )
        logger.info(funnel_summary)
        print(funnel_summary)


def _patch_fastwam_pad_before_config_load() -> None:
    """Apply channel-pad hooks before RolloutConfig parses the policy JSON."""
    if not _env_flag("LAB01_FASTWAM_PAD_ACTION"):
        return
    from simstudio.policies.fastwam_pad import patch_fastwam_pad_action_channels

    patch_fastwam_pad_action_channels(int(os.environ.get("LAB01_FASTWAM_REAL_ACTION_DIM", "6")))


def _patch_groot_processor_before_config_load() -> None:
    """Use public Qwen3-VL tokenizer files when Cosmos-Reason2-2B is gated."""
    argv = " ".join(sys.argv).lower()
    if not (
        _env_flag("LAB01_GROOT")
        or "groot" in argv
        or "groot" in os.environ.get("LAB01_EVAL_CONFIG", "").lower()
        or "groot" in os.environ.get("LAB01_POLICY_PATH", "").lower()
    ):
        return
    from simstudio.policies.groot_n17 import patch_groot_n17_vlm_processor_source

    patch_groot_n17_vlm_processor_source()


def main() -> None:
    register_third_party_plugins()
    init_logging()
    _patch_fastwam_pad_before_config_load()
    _patch_groot_processor_before_config_load()
    eval_main()


if __name__ == "__main__":
    _prepare_argv_for_rollout_parser()
    main()
