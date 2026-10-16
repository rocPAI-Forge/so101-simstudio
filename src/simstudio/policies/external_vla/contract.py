"""Versioned, model-neutral contracts for external VLA inference.

The wire contract deliberately uses named SO-101 joints and camera views. Model
adapters may use padded tensors, relative actions, or end-effector commands
internally, but they must return absolute SO-101 joint positions in radians.
"""

from __future__ import annotations

import base64
import io
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from simstudio.common.constants import JOINT_NAMES

PROTOCOL_VERSION = "1.0"
CANONICAL_CAMERA_NAMES = ("camera_front", "camera_top", "camera_wrist")
CANONICAL_JOINT_NAMES = tuple(JOINT_NAMES)


class ContractError(ValueError):
    """Raised when a VLA request or response violates the canonical contract."""


def _require_protocol(version: str) -> None:
    if version != PROTOCOL_VERSION:
        raise ContractError(f"Unsupported protocol version {version!r}; expected {PROTOCOL_VERSION!r}")


def _finite_float(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{field_name} must be finite, got {value!r}")
    return result


@dataclass(frozen=True)
class ObservationFrame:
    """One canonical SO-101 observation sent to a VLA adapter."""

    task: str
    joint_positions: Mapping[str, float]
    images: Mapping[str, np.ndarray]
    timestamp_ns: int = field(default_factory=time.time_ns)
    joint_velocities: Mapping[str, float] | None = None
    end_effector_xyz: tuple[float, float, float] | None = None
    protocol_version: str = PROTOCOL_VERSION

    def validate(
        self,
        *,
        required_cameras: tuple[str, ...] = (),
        required_state_fields: tuple[str, ...] = ("joint_positions",),
    ) -> None:
        _require_protocol(self.protocol_version)
        if not self.task.strip():
            raise ContractError("task must be a non-empty string")
        if self.timestamp_ns <= 0:
            raise ContractError("timestamp_ns must be positive")
        if tuple(self.joint_positions) != CANONICAL_JOINT_NAMES:
            raise ContractError(
                "joint_positions must use canonical order "
                f"{CANONICAL_JOINT_NAMES}, got {tuple(self.joint_positions)}"
            )
        for name, value in self.joint_positions.items():
            _finite_float(value, f"joint_positions.{name}")
        if "joint_velocities" in required_state_fields and self.joint_velocities is None:
            raise ContractError("Adapter requires joint_velocities")
        if self.joint_velocities is not None:
            if tuple(self.joint_velocities) != CANONICAL_JOINT_NAMES:
                raise ContractError("joint_velocities must use canonical joint order")
            for name, value in self.joint_velocities.items():
                _finite_float(value, f"joint_velocities.{name}")
        if "end_effector_xyz" in required_state_fields and self.end_effector_xyz is None:
            raise ContractError("Adapter requires end_effector_xyz")
        if self.end_effector_xyz is not None:
            if len(self.end_effector_xyz) != 3:
                raise ContractError("end_effector_xyz must contain exactly three values")
            for i, value in enumerate(self.end_effector_xyz):
                _finite_float(value, f"end_effector_xyz[{i}]")
        missing = set(required_cameras) - set(self.images)
        if missing:
            raise ContractError(f"Missing required cameras: {sorted(missing)}")
        for name, image in self.images.items():
            if name not in CANONICAL_CAMERA_NAMES:
                raise ContractError(f"Unknown canonical camera {name!r}")
            array = np.asarray(image)
            if array.ndim != 3 or array.shape[-1] != 3:
                raise ContractError(f"{name} must be HxWx3, got {array.shape}")
            if array.dtype != np.uint8:
                raise ContractError(f"{name} must be uint8, got {array.dtype}")

    def to_wire(self, *, jpeg_quality: int = 90) -> dict[str, Any]:
        self.validate()
        if not 1 <= jpeg_quality <= 100:
            raise ContractError("jpeg_quality must be in [1, 100]")
        encoded_images: dict[str, dict[str, Any]] = {}
        for name, array in self.images.items():
            output = io.BytesIO()
            Image.fromarray(np.asarray(array), mode="RGB").save(output, format="JPEG", quality=jpeg_quality)
            encoded_images[name] = {
                "encoding": "jpeg",
                "data": base64.b64encode(output.getvalue()).decode("ascii"),
                "shape": list(array.shape),
            }
        return {
            "protocol_version": self.protocol_version,
            "timestamp_ns": self.timestamp_ns,
            "task": self.task,
            "joint_positions": dict(self.joint_positions),
            "joint_velocities": (dict(self.joint_velocities) if self.joint_velocities is not None else None),
            "end_effector_xyz": (list(self.end_effector_xyz) if self.end_effector_xyz is not None else None),
            "images": encoded_images,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> ObservationFrame:
        images: dict[str, np.ndarray] = {}
        for name, encoded in payload.get("images", {}).items():
            if encoded.get("encoding") != "jpeg":
                raise ContractError(f"Unsupported image encoding for {name}: {encoded.get('encoding')!r}")
            raw = base64.b64decode(encoded["data"], validate=True)
            with Image.open(io.BytesIO(raw)) as image:
                images[name] = np.asarray(image.convert("RGB"), dtype=np.uint8)
            declared_shape = tuple(int(value) for value in encoded.get("shape", ()))
            if declared_shape and images[name].shape != declared_shape:
                raise ContractError(
                    f"Decoded {name} shape {images[name].shape} does not match {declared_shape}"
                )
        ee = payload.get("end_effector_xyz")
        result = cls(
            protocol_version=str(payload.get("protocol_version", "")),
            timestamp_ns=int(payload["timestamp_ns"]),
            task=str(payload["task"]),
            joint_positions={name: float(value) for name, value in payload["joint_positions"].items()},
            joint_velocities=(
                {name: float(value) for name, value in payload["joint_velocities"].items()}
                if payload.get("joint_velocities") is not None
                else None
            ),
            end_effector_xyz=tuple(float(value) for value in ee) if ee is not None else None,
            images=images,
        )
        result.validate()
        return result


@dataclass(frozen=True)
class ActionChunk:
    """Canonical absolute joint-position chunk returned by every adapter."""

    actions: np.ndarray
    source_timestamp_ns: int
    dt_s: float
    generated_at_ns: int = field(default_factory=time.time_ns)
    joint_names: tuple[str, ...] = CANONICAL_JOINT_NAMES
    protocol_version: str = PROTOCOL_VERSION
    timings_ms: Mapping[str, float] = field(default_factory=dict)

    def validate(
        self,
        *,
        expected_horizon: int | None = None,
        joint_limits: Mapping[str, tuple[float, float]] | None = None,
        stale_after_s: float | None = None,
        now_ns: int | None = None,
    ) -> None:
        _require_protocol(self.protocol_version)
        if self.joint_names != CANONICAL_JOINT_NAMES:
            raise ContractError(f"Action joint order must be {CANONICAL_JOINT_NAMES}, got {self.joint_names}")
        action = np.asarray(self.actions)
        if action.ndim != 2 or action.shape[1] != len(CANONICAL_JOINT_NAMES):
            raise ContractError(f"actions must have shape [T, 6], got {action.shape}")
        if action.shape[0] == 0:
            raise ContractError("actions must contain at least one timestep")
        if expected_horizon is not None and action.shape[0] != expected_horizon:
            raise ContractError(f"Expected action horizon {expected_horizon}, got {action.shape[0]}")
        if not np.isfinite(action).all():
            raise ContractError("actions contain NaN or infinite values")
        if self.source_timestamp_ns <= 0 or self.generated_at_ns <= 0:
            raise ContractError("source_timestamp_ns and generated_at_ns must be positive")
        if not math.isfinite(self.dt_s) or self.dt_s <= 0:
            raise ContractError(f"dt_s must be positive and finite, got {self.dt_s!r}")
        if stale_after_s is not None:
            age_s = ((now_ns or time.time_ns()) - self.generated_at_ns) / 1e9
            if age_s > stale_after_s:
                raise ContractError(f"Action chunk is stale ({age_s:.3f}s > {stale_after_s:.3f}s)")
        if joint_limits:
            for column, name in enumerate(self.joint_names):
                if name not in joint_limits:
                    raise ContractError(f"Missing joint limit for {name}")
                low, high = joint_limits[name]
                values = action[:, column]
                if np.any(values < low) or np.any(values > high):
                    raise ContractError(f"Action for {name} is outside [{low}, {high}]")
        for name, value in self.timings_ms.items():
            if _finite_float(value, f"timings_ms.{name}") < 0:
                raise ContractError(f"timings_ms.{name} cannot be negative")

    def to_wire(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "source_timestamp_ns": self.source_timestamp_ns,
            "generated_at_ns": self.generated_at_ns,
            "dt_s": self.dt_s,
            "joint_names": list(self.joint_names),
            "actions": np.asarray(self.actions, dtype=np.float32).tolist(),
            "timings_ms": dict(self.timings_ms),
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> ActionChunk:
        result = cls(
            protocol_version=str(payload.get("protocol_version", "")),
            source_timestamp_ns=int(payload["source_timestamp_ns"]),
            generated_at_ns=int(payload["generated_at_ns"]),
            dt_s=float(payload["dt_s"]),
            joint_names=tuple(payload["joint_names"]),
            actions=np.asarray(payload["actions"], dtype=np.float32),
            timings_ms={str(name): float(value) for name, value in payload.get("timings_ms", {}).items()},
        )
        result.validate()
        return result


@dataclass(frozen=True)
class Capabilities:
    """Adapter requirements negotiated before the first episode."""

    adapter_name: str
    required_cameras: tuple[str, ...]
    required_state_fields: tuple[str, ...]
    control_fps: float
    action_horizon: int
    execution_horizon: int
    supports_history: bool = False
    protocol_version: str = PROTOCOL_VERSION

    def validate(self) -> None:
        _require_protocol(self.protocol_version)
        unknown = set(self.required_cameras) - set(CANONICAL_CAMERA_NAMES)
        if unknown:
            raise ContractError(f"Capabilities contain unknown cameras: {sorted(unknown)}")
        valid_state = {"joint_positions", "joint_velocities", "end_effector_xyz"}
        unknown_state = set(self.required_state_fields) - valid_state
        if unknown_state:
            raise ContractError(f"Capabilities contain unknown state fields: {sorted(unknown_state)}")
        if not math.isfinite(self.control_fps) or self.control_fps <= 0:
            raise ContractError("control_fps must be positive")
        if self.action_horizon <= 0:
            raise ContractError("action_horizon must be positive")
        if not 0 < self.execution_horizon <= self.action_horizon:
            raise ContractError("execution_horizon must be in [1, action_horizon]")

    def to_wire(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": self.protocol_version,
            "adapter_name": self.adapter_name,
            "required_cameras": list(self.required_cameras),
            "required_state_fields": list(self.required_state_fields),
            "control_fps": self.control_fps,
            "action_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "supports_history": self.supports_history,
        }

    @classmethod
    def from_wire(cls, payload: Mapping[str, Any]) -> Capabilities:
        result = cls(
            protocol_version=str(payload.get("protocol_version", "")),
            adapter_name=str(payload["adapter_name"]),
            required_cameras=tuple(payload["required_cameras"]),
            required_state_fields=tuple(payload["required_state_fields"]),
            control_fps=float(payload["control_fps"]),
            action_horizon=int(payload["action_horizon"]),
            execution_horizon=int(payload["execution_horizon"]),
            supports_history=bool(payload.get("supports_history", False)),
        )
        result.validate()
        return result
