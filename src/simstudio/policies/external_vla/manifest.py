"""Deployment manifest for reproducible external VLA inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contract import (
    CANONICAL_CAMERA_NAMES,
    CANONICAL_JOINT_NAMES,
    PROTOCOL_VERSION,
    Capabilities,
    ContractError,
)

MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DeploymentManifest:
    """Everything inference needs beyond model weights.

    A training pipeline may be completely independent from LeRobot, but its
    deployable output is incomplete until it provides this manifest and the
    referenced normalization/processor artifacts.
    """

    adapter: str
    model_id: str
    model_revision: str
    required_cameras: tuple[str, ...]
    required_state_fields: tuple[str, ...] = ("joint_positions",)
    control_fps: float = 20.0
    action_horizon: int = 30
    execution_horizon: int = 10
    action_semantics: str = "absolute_joint_position"
    action_units: str = "radian"
    joint_names: tuple[str, ...] = CANONICAL_JOINT_NAMES
    adapter_config: Mapping[str, Any] = field(default_factory=dict)
    normalization: Mapping[str, Any] = field(default_factory=dict)
    joint_limits: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    protocol_version: str = PROTOCOL_VERSION
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise ContractError(
                f"Unsupported manifest schema {self.schema_version}; expected {MANIFEST_SCHEMA_VERSION}"
            )
        if not self.adapter or ":" not in self.adapter:
            raise ContractError("adapter must be an import target like 'package.module:ClassName'")
        if not self.model_id.strip():
            raise ContractError("model_id must be non-empty")
        if not self.model_revision.strip():
            raise ContractError("model_revision must pin a tag, commit, or local artifact version")
        if self.action_semantics != "absolute_joint_position":
            raise ContractError("Adapter boundary only accepts action_semantics='absolute_joint_position'")
        if self.action_units != "radian":
            raise ContractError("Adapter boundary only accepts action_units='radian'")
        if self.joint_names != CANONICAL_JOINT_NAMES:
            raise ContractError(f"joint_names must be {CANONICAL_JOINT_NAMES}")
        unknown_cameras = set(self.required_cameras) - set(CANONICAL_CAMERA_NAMES)
        if unknown_cameras:
            raise ContractError(f"Unknown required cameras: {sorted(unknown_cameras)}")
        self.capabilities().validate()
        if set(self.joint_limits) != set(CANONICAL_JOINT_NAMES):
            raise ContractError("joint_limits must contain all canonical joints")
        for name, bounds in self.joint_limits.items():
            if len(bounds) != 2 or float(bounds[0]) >= float(bounds[1]):
                raise ContractError(f"Invalid joint_limits for {name}: {bounds}")

    def capabilities(self) -> Capabilities:
        return Capabilities(
            adapter_name=self.adapter,
            required_cameras=self.required_cameras,
            required_state_fields=self.required_state_fields,
            control_fps=self.control_fps,
            action_horizon=self.action_horizon,
            execution_horizon=self.execution_horizon,
            supports_history=bool(self.adapter_config.get("supports_history", False)),
            protocol_version=self.protocol_version,
        )

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "adapter": self.adapter,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "required_cameras": list(self.required_cameras),
            "required_state_fields": list(self.required_state_fields),
            "control_fps": self.control_fps,
            "action_horizon": self.action_horizon,
            "execution_horizon": self.execution_horizon,
            "action_semantics": self.action_semantics,
            "action_units": self.action_units,
            "joint_names": list(self.joint_names),
            "adapter_config": dict(self.adapter_config),
            "normalization": dict(self.normalization),
            "joint_limits": {
                name: [float(bounds[0]), float(bounds[1])] for name, bounds in self.joint_limits.items()
            },
        }

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeploymentManifest:
        manifest = cls(
            schema_version=int(payload.get("schema_version", 0)),
            protocol_version=str(payload.get("protocol_version", "")),
            adapter=str(payload["adapter"]),
            model_id=str(payload["model_id"]),
            model_revision=str(payload["model_revision"]),
            required_cameras=tuple(payload["required_cameras"]),
            required_state_fields=tuple(payload.get("required_state_fields", ("joint_positions",))),
            control_fps=float(payload.get("control_fps", 20.0)),
            action_horizon=int(payload["action_horizon"]),
            execution_horizon=int(payload["execution_horizon"]),
            action_semantics=str(payload.get("action_semantics", "")),
            action_units=str(payload.get("action_units", "")),
            joint_names=tuple(payload["joint_names"]),
            adapter_config=dict(payload.get("adapter_config", {})),
            normalization=dict(payload.get("normalization", {})),
            joint_limits={
                str(name): (float(bounds[0]), float(bounds[1]))
                for name, bounds in payload.get("joint_limits", {}).items()
            },
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, path: str | Path) -> DeploymentManifest:
        source = Path(path)
        with source.open(encoding="utf-8") as stream:
            return cls.from_dict(json.load(stream))
