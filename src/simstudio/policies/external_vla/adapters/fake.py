"""Deterministic adapter used for contract and closed-loop smoke tests."""

from __future__ import annotations

import time

import numpy as np

from ..contract import ActionChunk, Capabilities, ObservationFrame
from ..manifest import DeploymentManifest


class FakeAdapter:
    """Repeat the current pose, optionally adding a configured per-step delta."""

    def __init__(self, manifest: DeploymentManifest):
        self.manifest = manifest
        self._capabilities = manifest.capabilities()
        delta = manifest.adapter_config.get("joint_delta_per_step", [0.0] * 6)
        self._delta = np.asarray(delta, dtype=np.float32)
        if self._delta.shape != (6,):
            raise ValueError("fake joint_delta_per_step must contain six values")
        self._episode_id: str | None = None

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def reset_episode(self, episode_id: str | None = None) -> None:
        self._episode_id = episode_id

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        observation.validate(
            required_cameras=self.capabilities.required_cameras,
            required_state_fields=self.capabilities.required_state_fields,
        )
        start = np.asarray(list(observation.joint_positions.values()), dtype=np.float32)
        steps = np.arange(1, self.capabilities.action_horizon + 1, dtype=np.float32)[:, None]
        actions = start[None, :] + steps * self._delta[None, :]
        return ActionChunk(
            actions=actions,
            source_timestamp_ns=observation.timestamp_ns,
            generated_at_ns=time.time_ns(),
            dt_s=1.0 / self.capabilities.control_fps,
            timings_ms={"model": 0.0},
        )

    def close(self) -> None:
        pass
