"""Xiaomi XR-1 sidecar adapter with explicit single-arm and IK hooks.

XR-1's native release targets a dual-arm mobile manipulator and returns
end-effector actions. The deployment environment must provide a small runtime
factory and an IK solver; this module owns their conversion to the canonical
SO-101 joint-position chunk.
"""

from __future__ import annotations

import importlib
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..contract import CANONICAL_JOINT_NAMES, ActionChunk, Capabilities, ObservationFrame
from ..manifest import DeploymentManifest


def _load_symbol(target: str) -> Any:
    if ":" not in target:
        raise ValueError(f"Import target must be 'module:symbol', got {target!r}")
    module_name, symbol_name = target.split(":", 1)
    return getattr(importlib.import_module(module_name), symbol_name)


class XiaomiXR1Adapter:
    """Bridge an upstream XR-1 runtime client to one SO-101 arm."""

    def __init__(self, manifest: DeploymentManifest):
        self.manifest = manifest
        self._capabilities = manifest.capabilities()
        config = dict(manifest.adapter_config)
        if not bool(config.get("native_runtime_handles_normalization", False)):
            raise ValueError(
                "XR-1 manifest must affirm native_runtime_handles_normalization=true; "
                "the released client and training stats must stay paired"
            )
        self._camera_mapping = dict(config["camera_mapping"])
        if set(self._camera_mapping) != set(self.capabilities.required_cameras):
            raise ValueError("camera_mapping must map every required canonical camera")
        if set(self._camera_mapping.values()) != {"ego", "wrist_left", "wrist_right"}:
            raise ValueError("camera_mapping targets must be ego, wrist_left, and wrist_right")
        arm_slot = str(config.get("arm_slot", "right"))
        if arm_slot not in {"left", "right"}:
            raise ValueError("arm_slot must be 'left' or 'right'")
        self._arm_slot = arm_slot
        runtime_factory = _load_symbol(str(config["runtime_factory"]))
        self._runtime = runtime_factory(
            model_id=manifest.model_id,
            model_revision=manifest.model_revision,
            **dict(config.get("runtime_kwargs", {})),
        )
        self._ik_solver: Callable[..., Sequence[Sequence[float]]] = _load_symbol(str(config["ik_solver"]))
        self._action_component_key = str(config.get("action_component_key", f"{arm_slot}_ee"))

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def reset_episode(self, episode_id: str | None = None) -> None:
        reset = getattr(self._runtime, "reset_episode", None)
        if callable(reset):
            reset(episode_id)

    def _robot_state(self, observation: ObservationFrame) -> dict[str, Any]:
        joints = list(observation.joint_positions.values())
        empty_joint = [0.0] * min(7, len(joints))
        state = {
            "left_arm_joint": joints if self._arm_slot == "left" else empty_joint,
            "left_gripper_pos": joints[-1] if self._arm_slot == "left" else 0.0,
            "left_ee_pos": [0.0, 0.0, 0.0],
            "left_ee_rotm": np.eye(3, dtype=np.float32).reshape(-1).tolist(),
            "right_arm_joint": joints if self._arm_slot == "right" else empty_joint,
            "right_gripper_pos": joints[-1] if self._arm_slot == "right" else 0.0,
            "right_ee_pos": [0.0, 0.0, 0.0],
            "right_ee_rotm": np.eye(3, dtype=np.float32).reshape(-1).tolist(),
        }
        if observation.end_effector_xyz is not None:
            state[f"{self._arm_slot}_ee_pos"] = list(observation.end_effector_xyz)
        return state

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        observation.validate(
            required_cameras=self.capabilities.required_cameras,
            required_state_fields=self.capabilities.required_state_fields,
        )
        native_images = {
            native_name: observation.images[canonical_name]
            for canonical_name, native_name in self._camera_mapping.items()
        }
        started = time.perf_counter()
        result = self._runtime.predict(
            images=native_images,
            instruction=observation.task,
            robot_state=self._robot_state(observation),
        )
        if not isinstance(result, Mapping):
            raise TypeError("XR-1 runtime predict() must return a mapping")
        components = result.get("action_components", result)
        if self._action_component_key not in components:
            raise KeyError(f"XR-1 response lacks action component {self._action_component_key!r}")
        current = np.asarray(list(observation.joint_positions.values()), dtype=np.float32)
        actions = np.asarray(
            self._ik_solver(
                current_joints=current,
                action_components=components,
                arm_slot=self._arm_slot,
                horizon=self.capabilities.action_horizon,
                dt_s=1.0 / self.capabilities.control_fps,
            ),
            dtype=np.float32,
        )
        model_ms = (time.perf_counter() - started) * 1000.0
        if actions.shape != (self.capabilities.action_horizon, len(CANONICAL_JOINT_NAMES)):
            raise ValueError(
                f"XR-1 IK solver must return ({self.capabilities.action_horizon}, 6), got {actions.shape}"
            )
        return ActionChunk(
            actions=actions,
            source_timestamp_ns=observation.timestamp_ns,
            dt_s=1.0 / self.capabilities.control_fps,
            timings_ms={"model_and_postprocess": model_ms},
        )

    def close(self) -> None:
        close = getattr(self._runtime, "close", None)
        if callable(close):
            close()
