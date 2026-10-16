"""LeRobot plugin for SimStudio external VLA inference."""

try:
    import lerobot  # noqa: F401
except ImportError as exc:  # pragma: no cover - import guard for standalone inspection
    raise ImportError("lerobot is required by lerobot_policy_external_vla") from exc

from .configuration_external_vla import ExternalVLAConfig
from .modeling_external_vla import ExternalVLAPolicy
from .processor_external_vla import make_external_vla_pre_post_processors

__all__ = [
    "ExternalVLAConfig",
    "ExternalVLAPolicy",
    "make_external_vla_pre_post_processors",
]
