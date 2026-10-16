"""Adapter interface and dynamic loader."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from ..contract import ActionChunk, Capabilities, ObservationFrame
from ..manifest import DeploymentManifest


@runtime_checkable
class VLAAdapter(Protocol):
    """Model-specific code hosted in-process or by a sidecar."""

    @property
    def capabilities(self) -> Capabilities: ...

    def reset_episode(self, episode_id: str | None = None) -> None: ...

    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk: ...

    def close(self) -> None: ...


def load_adapter(manifest: DeploymentManifest) -> VLAAdapter:
    """Instantiate the ``module:Class`` named by a deployment manifest."""
    module_name, class_name = manifest.adapter.split(":", 1)
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name)
    adapter = adapter_class(manifest)
    if not isinstance(adapter, VLAAdapter):
        raise TypeError(f"{manifest.adapter} does not implement VLAAdapter")
    if adapter.capabilities != manifest.capabilities():
        raise ValueError(
            f"{manifest.adapter} capabilities do not match deployment manifest: "
            f"{adapter.capabilities!r} != {manifest.capabilities()!r}"
        )
    return adapter
