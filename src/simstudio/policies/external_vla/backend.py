"""Inference backends and local response safety checks."""

from __future__ import annotations

import time
from typing import Protocol, runtime_checkable

from .adapters.base import VLAAdapter, load_adapter
from .contract import ActionChunk, Capabilities, ObservationFrame
from .manifest import DeploymentManifest


@runtime_checkable
class VLABackend(Protocol):
    @property
    def capabilities(self) -> Capabilities: ...

    def health(self) -> dict[str, object]: ...

    def reset_episode(self, episode_id: str | None = None) -> None: ...

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk: ...

    def close(self) -> None: ...


class SafeBackend:
    """Validate requests and responses at the local trust boundary."""

    def __init__(
        self,
        backend: VLABackend,
        manifest: DeploymentManifest,
        *,
        stale_after_s: float,
    ):
        self._backend = backend
        self._manifest = manifest
        self._stale_after_s = stale_after_s
        manifest.validate()
        if backend.capabilities != manifest.capabilities():
            raise ValueError("Backend capabilities do not match deployment manifest")
        if stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")

    @property
    def capabilities(self) -> Capabilities:
        return self._backend.capabilities

    def health(self) -> dict[str, object]:
        return self._backend.health()

    def reset_episode(self, episode_id: str | None = None) -> None:
        self._backend.reset_episode(episode_id)

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        observation.validate(
            required_cameras=self.capabilities.required_cameras,
            required_state_fields=self.capabilities.required_state_fields,
        )
        chunk = self._backend.predict_chunk(observation)
        if chunk.source_timestamp_ns != observation.timestamp_ns:
            raise ValueError("Backend response source_timestamp_ns does not match the request observation")
        chunk.validate(
            expected_horizon=self.capabilities.action_horizon,
            joint_limits=self._manifest.joint_limits or None,
            stale_after_s=self._stale_after_s,
        )
        return chunk

    def close(self) -> None:
        self._backend.close()


class InProcessBackend:
    """Run a model adapter in the caller's Python environment."""

    def __init__(self, manifest: DeploymentManifest, adapter: VLAAdapter | None = None):
        self._adapter = adapter or load_adapter(manifest)

    @property
    def capabilities(self) -> Capabilities:
        return self._adapter.capabilities

    def health(self) -> dict[str, object]:
        return {
            "ok": True,
            "adapter": self.capabilities.adapter_name,
            "time_ns": time.time_ns(),
        }

    def reset_episode(self, episode_id: str | None = None) -> None:
        self._adapter.reset_episode(episode_id)

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        return self._adapter.predict_chunk(observation)

    def close(self) -> None:
        self._adapter.close()
