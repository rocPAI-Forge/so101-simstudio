"""Action mapping utilities.

Convert normalized teleoperator velocity commands into robot-native action dicts.
"""

from typing import Any

from simstudio.common.constants import JOINT_NAMES


def velocity_to_position_action(
    velocities: dict[str, float],
    current_positions: dict[str, float],
    dt: float,
) -> dict[str, float]:
    """Integrate velocity commands to target joint positions.

    Used for real robots where the teleop output must be converted to
    position targets before sending.
    """
    action = {}
    for joint in JOINT_NAMES:
        pos = current_positions.get(joint, 0.0)
        vel_key = _joint_to_velocity_key(joint)
        vel = velocities.get(vel_key, 0.0)
        action[f"{joint}.pos"] = pos + vel * dt
    return action


def _joint_to_velocity_key(joint_name: str) -> str:
    mapping = {
        "shoulder_pan": "vx",
        "shoulder_lift": "vy",
        "elbow_flex": "vz",
        "wrist_flex": "wrist_flex_rate",
        "wrist_roll": "yaw_rate",
        "gripper": "gripper_delta",
    }
    return mapping[joint_name]


def identity_action(velocities: dict[str, float]) -> dict[str, Any]:
    """Pass teleop velocities through unchanged.

    Used by MuJoCo robot which interprets velocities internally.
    """
    return dict(velocities)
