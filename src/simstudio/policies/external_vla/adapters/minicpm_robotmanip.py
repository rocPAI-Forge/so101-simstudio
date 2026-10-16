"""MiniCPM-RobotManip reference adapter.

The public checkpoint emits padded 80-D actions but does not publish a
canonical SO-101 embodiment mapping. This adapter therefore fails closed until
the deployment manifest explicitly supplies channel indices and affine
normalization for both state and action.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from PIL import Image

from ..contract import CANONICAL_JOINT_NAMES, ActionChunk, Capabilities, ObservationFrame
from ..manifest import DeploymentManifest


def _ordered_mapping(config: dict[str, Any], key: str) -> list[int]:
    mapping = config.get(key)
    if not isinstance(mapping, dict) or set(mapping) != set(CANONICAL_JOINT_NAMES):
        raise ValueError(f"{key} must map every canonical joint name to a channel index")
    values = [int(mapping[name]) for name in CANONICAL_JOINT_NAMES]
    if len(set(values)) != len(values) or any(value < 0 or value >= 80 for value in values):
        raise ValueError(f"{key} indices must be six unique values in [0, 79]")
    return values


def _six_values(config: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(config.get(key), dtype=np.float32)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError(f"normalization.{key} must contain six finite values")
    return value


class MiniCPMRobotManipAdapter:
    """Load the public Transformers checkpoint and expose canonical actions."""

    STATE_DIM = 80

    def __init__(self, manifest: DeploymentManifest):
        self.manifest = manifest
        self._capabilities = manifest.capabilities()
        config = dict(manifest.adapter_config)
        normalization = dict(manifest.normalization)
        self._state_indices = _ordered_mapping(config, "state_indices")
        self._action_indices = _ordered_mapping(config, "action_indices")
        self._state_offset = _six_values(normalization, "state_offset")
        self._state_scale = _six_values(normalization, "state_scale")
        self._action_offset = _six_values(normalization, "action_offset")
        self._action_scale = _six_values(normalization, "action_scale")
        if np.any(self._state_scale == 0) or np.any(self._action_scale == 0):
            raise ValueError("state_scale and action_scale cannot contain zero")
        self._embodiment_id = int(config["embodiment_id"])
        self._prompt_template = str(config.get("prompt_template", "{task}"))
        self._device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "MiniCPM adapter requires its native transformers environment "
                "(the release model card pins transformers==5.7.0)"
            ) from exc
        self.processor = AutoProcessor.from_pretrained(
            manifest.model_id,
            revision=manifest.model_revision,
            trust_remote_code=True,
        )
        self.model = AutoModel.from_pretrained(
            manifest.model_id,
            revision=manifest.model_revision,
            trust_remote_code=True,
        )
        self.model.to(self._device).eval()
        if not 0 <= self._embodiment_id < self.model.action_head.max_num_embodiments:
            raise ValueError("embodiment_id is outside the checkpoint's supported range")

    @property
    def capabilities(self) -> Capabilities:
        return self._capabilities

    def reset_episode(self, episode_id: str | None = None) -> None:
        del episode_id

    def _state(self, observation: ObservationFrame) -> torch.Tensor:
        canonical = np.asarray(list(observation.joint_positions.values()), dtype=np.float32)
        normalized = (canonical - self._state_offset) / self._state_scale
        state = np.zeros(self.STATE_DIM, dtype=np.float32)
        state[self._state_indices] = normalized
        return torch.from_numpy(state).to(self._device).view(1, 1, self.STATE_DIM)

    def _vlm_inputs(self, observation: ObservationFrame) -> dict[str, torch.Tensor]:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": Image.fromarray(observation.images[name])}
            for name in self.capabilities.required_cameras
        ]
        content.append({"type": "text", "text": self._prompt_template.format(task=observation.task)})
        inputs = self.processor.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": False},
        )
        return {
            key: value.to(self._device) for key, value in inputs.items() if isinstance(value, torch.Tensor)
        }

    @torch.inference_mode()
    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk:
        observation.validate(
            required_cameras=self.capabilities.required_cameras,
            required_state_fields=self.capabilities.required_state_fields,
        )
        started = time.perf_counter()
        actions = self.model.predict_action(
            state=self._state(observation),
            embodiment_id=torch.tensor([self._embodiment_id], device=self._device),
            **self._vlm_inputs(observation),
        )[0]
        model_ms = (time.perf_counter() - started) * 1000.0
        selected = actions[:, self._action_indices].float().cpu().numpy()
        canonical = selected * self._action_scale + self._action_offset
        return ActionChunk(
            actions=canonical,
            source_timestamp_ns=observation.timestamp_ns,
            dt_s=1.0 / self.capabilities.control_fps,
            timings_ms={"model_and_postprocess": model_ms},
        )

    def close(self) -> None:
        del self.model
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
