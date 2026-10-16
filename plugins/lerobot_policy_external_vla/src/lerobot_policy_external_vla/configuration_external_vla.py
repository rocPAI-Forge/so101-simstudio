"""LeRobot configuration for the external VLA inference-only policy."""

from __future__ import annotations

from dataclasses import dataclass

from lerobot.configs import PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@PreTrainedConfig.register_subclass("external_vla")
@dataclass
class ExternalVLAConfig(PreTrainedConfig):
    """Configuration stored next to an external VLA deployment manifest."""

    manifest_path: str = "deployment_manifest.json"
    backend: str = "sidecar"
    endpoint: str = "http://127.0.0.1:10086"
    request_timeout_s: float = 10.0
    stale_after_s: float = 2.0
    jpeg_quality: int = 90
    chunk_size: int = 30
    n_action_steps: int = 10
    state_joint_names: tuple[str, ...] = (
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    )
    device: str | None = "cpu"

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.backend not in {"inprocess", "sidecar"}:
            raise ValueError("backend must be 'inprocess' or 'sidecar'")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0 < self.n_action_steps <= self.chunk_size:
            raise ValueError("n_action_steps must be in [1, chunk_size]")
        if self.request_timeout_s <= 0 or self.stale_after_s <= 0:
            raise ValueError("request_timeout_s and stale_after_s must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")

    def validate_features(self) -> None:
        state = self.robot_state_feature
        if state is None or state.shape[0] < 6:
            raise ValueError("ExternalVLA requires observation.state with at least six joint positions")
        action = self.action_feature
        if action is None or action.shape != (6,):
            raise ValueError("ExternalVLA output action must have shape (6,)")
        if not self.image_features:
            raise ValueError("ExternalVLA requires at least one image feature")
        if OBS_STATE not in (self.input_features or {}):
            raise ValueError(f"ExternalVLA requires {OBS_STATE!r}")
        if ACTION not in (self.output_features or {}):
            raise ValueError(f"ExternalVLA requires {ACTION!r}")

    def get_optimizer_preset(self) -> AdamWConfig:
        # The bridge is inference-only. This satisfies LeRobot's config contract
        # while ``forward`` gives a direct error if lerobot-train is attempted.
        return AdamWConfig(lr=0.0, weight_decay=0.0)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
