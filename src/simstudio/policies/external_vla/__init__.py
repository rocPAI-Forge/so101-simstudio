"""Model-neutral inference boundary for VLAs not built into LeRobot."""

from .backend import InProcessBackend, SafeBackend, VLABackend
from .contract import (
    CANONICAL_CAMERA_NAMES,
    CANONICAL_JOINT_NAMES,
    PROTOCOL_VERSION,
    ActionChunk,
    Capabilities,
    ContractError,
    ObservationFrame,
)
from .manifest import DeploymentManifest
from .sidecar import SidecarClient

__all__ = [
    "PROTOCOL_VERSION",
    "CANONICAL_CAMERA_NAMES",
    "CANONICAL_JOINT_NAMES",
    "ActionChunk",
    "Capabilities",
    "ContractError",
    "DeploymentManifest",
    "InProcessBackend",
    "ObservationFrame",
    "SafeBackend",
    "SidecarClient",
    "VLABackend",
]
