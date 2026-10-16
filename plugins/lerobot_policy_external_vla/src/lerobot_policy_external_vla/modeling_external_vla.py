"""LeRobot policy proxy for in-process and sidecar VLA adapters."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from lerobot.policies import PreTrainedPolicy
from lerobot.utils.constants import OBS_STATE
from torch import Tensor, nn

from simstudio.policies.external_vla import (
    ActionChunk,
    DeploymentManifest,
    InProcessBackend,
    ObservationFrame,
    SafeBackend,
    SidecarClient,
)

from .configuration_external_vla import ExternalVLAConfig


class ExternalVLAPolicy(PreTrainedPolicy):
    """Inference-only proxy that preserves LeRobot rollout and RTC behavior."""

    config_class = ExternalVLAConfig
    name = "external_vla"

    def __init__(self, config: ExternalVLAConfig, **_: Any):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.register_parameter("_checkpoint_marker", nn.Parameter(torch.zeros(()), requires_grad=False))
        self._manifest = DeploymentManifest.load(config.manifest_path)
        if self._manifest.action_horizon != config.chunk_size:
            raise ValueError(
                "Manifest action_horizon must equal policy chunk_size: "
                f"{self._manifest.action_horizon} != {config.chunk_size}"
            )
        if self._manifest.execution_horizon != config.n_action_steps:
            raise ValueError(
                "Manifest execution_horizon must equal policy n_action_steps: "
                f"{self._manifest.execution_horizon} != {config.n_action_steps}"
            )
        raw_backend = (
            InProcessBackend(self._manifest)
            if config.backend == "inprocess"
            else SidecarClient(
                config.endpoint,
                timeout_s=config.request_timeout_s,
                jpeg_quality=config.jpeg_quality,
            )
        )
        self._backend = SafeBackend(
            raw_backend,
            self._manifest,
            stale_after_s=config.stale_after_s,
        )
        self._action_queue: deque[Tensor] = deque()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: ExternalVLAConfig | None = None,
        revision: str | None = None,
        **kwargs: Any,
    ) -> ExternalVLAPolicy:
        """Load proxy config/manifest without requiring bridge model weights."""
        if config is None:
            config = ExternalVLAConfig.from_pretrained(
                pretrained_name_or_path,
                revision=revision,
                **kwargs,
            )
        root = Path(pretrained_name_or_path)
        manifest = Path(config.manifest_path)
        if not manifest.is_absolute():
            if root.is_dir():
                manifest = root / manifest
            else:
                manifest = Path(
                    hf_hub_download(
                        repo_id=str(pretrained_name_or_path),
                        filename=config.manifest_path,
                        revision=revision,
                    )
                )
        config.manifest_path = str(manifest)
        policy = cls(config)
        policy.eval()
        return policy

    def get_optim_params(self) -> dict[str, object]:
        return {"params": []}

    def reset(self) -> None:
        self._action_queue.clear()
        self._backend.reset_episode()

    def close(self) -> None:
        self._backend.close()

    @staticmethod
    def _first_sample(value: Any) -> np.ndarray:
        if isinstance(value, Tensor):
            array = value.detach().cpu().numpy()
        else:
            array = np.asarray(value)
        if array.ndim > 0 and array.shape[0] == 1:
            array = array[0]
        return array

    @classmethod
    def _image_to_uint8_hwc(cls, value: Any) -> np.ndarray:
        image = cls._first_sample(value)
        if image.ndim != 3:
            raise ValueError(f"Expected an image tensor with three dimensions, got {image.shape}")
        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))
        if image.shape[-1] != 3:
            raise ValueError(f"Expected RGB image, got {image.shape}")
        if np.issubdtype(image.dtype, np.floating):
            if image.size and float(np.nanmax(image)) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0.0, 255.0)
        return image.astype(np.uint8)

    def _to_observation(self, batch: dict[str, Any]) -> ObservationFrame:
        state = self._first_sample(batch[OBS_STATE]).reshape(-1)
        if state.shape[0] < 6:
            raise ValueError(f"observation.state needs at least six values, got {state.shape}")
        positions = {name: float(state[index]) for index, name in enumerate(self.config.state_joint_names)}
        velocities = None
        if "joint_velocities" in self._manifest.required_state_fields:
            if state.shape[0] < 12:
                raise ValueError(
                    "Manifest requires joint velocities but observation.state is shorter than 12"
                )
            velocities = {
                name: float(state[6 + index]) for index, name in enumerate(self.config.state_joint_names)
            }
        ee_xyz = None
        if "end_effector_xyz" in self._manifest.required_state_fields:
            if state.shape[0] < 15:
                raise ValueError("Manifest requires EE XYZ but observation.state is shorter than 15")
            ee_xyz = tuple(float(value) for value in state[12:15])
        images = {
            camera: self._image_to_uint8_hwc(batch[f"observation.images.{camera}"])
            for camera in self._manifest.required_cameras
        }
        task_value = batch.get("task", "")
        if isinstance(task_value, (list, tuple)):
            task_value = task_value[0] if task_value else ""
        return ObservationFrame(
            task=str(task_value),
            joint_positions=positions,
            joint_velocities=velocities,
            end_effector_xyz=ee_xyz,
            images=images,
            timestamp_ns=time.time_ns(),
        )

    @torch.inference_mode()
    def predict_action_chunk(self, batch: dict[str, Any], **_: Any) -> Tensor:
        observation = self._to_observation(batch)
        chunk: ActionChunk = self._backend.predict_chunk(observation)
        return torch.from_numpy(np.asarray(chunk.actions, dtype=np.float32)).unsqueeze(0)

    @torch.inference_mode()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        if not self._action_queue:
            actions = self.predict_action_chunk(batch, **kwargs)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    def forward(self, batch: dict[str, Tensor], **_: Any) -> tuple[Tensor, dict[str, Any]]:
        raise RuntimeError(
            "ExternalVLAPolicy is an inference bridge, not a training policy. "
            "Train the model in its native stack and export a deployment manifest."
        )
