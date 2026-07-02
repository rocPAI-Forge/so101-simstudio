"""Shared constants for SO-101 robots and teleoperators."""

# Joint names in standard order
JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Default SO-101 home configuration (radians)
DEFAULT_HOME_POSITION = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.3,
    "elbow_flex": 0.6,
    "wrist_flex": 1.2,
    "wrist_roll": 0.0,
    "gripper": 0.8,
}

# Normalized teleop velocity command keys
TELEOP_VELOCITY_KEYS = ["vx", "vy", "vz", "wrist_flex_rate", "yaw_rate", "gripper_delta"]
