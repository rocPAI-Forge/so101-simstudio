"""SO-101 real leader arm teleoperator.

Wraps LeRobot's SOLeader implementation for Feetech STS3215 motors.
Reads normalized positions and scales to MuJoCo radian targets.
"""

import os

from lerobot.teleoperators.so_leader.so_leader import SOLeader

from simstudio.teleoperators.so101_leader.config import SO101LeaderTeleopConfig

# Set SO101_GRIPPER_DEBUG=1 to print raw + scaled gripper values every frame.
_GRIPPER_DEBUG = os.environ.get("SO101_GRIPPER_DEBUG") == "1"

# MuJoCo joint ranges (from so101_new_calib.xml ctrlrange).
# NOTE: the gripper lower bound is deliberately kept above the hard ctrlrange
# floor (-0.17453). Driving the sim gripper all the way to the hard limit makes
# the two jaw geoms self-collide; the soft position servo (kp=17.8) cannot hold
# against the contact force and the solver flings the gripper wide open. Mapping
# "fully closed" to -0.1 stays visually closed without hitting that instability.
_GRIPPER_CLOSED_SAFE = -0.1
_MUJOCO_JOINT_RANGE = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (_GRIPPER_CLOSED_SAFE, 1.74533),
}

# Leader arm normalized ranges (from FeetechMotorsBus norm modes)
_LEADER_RANGE_ARM = (-100.0, 100.0)  # RANGE_M100_100 for arm joints
_LEADER_RANGE_GRIPPER = (0.0, 100.0)  # RANGE_0_100 for gripper


class SO101LeaderTeleop(SOLeader):
    """SO-101 leader arm teleoperator (Feetech STS3215, 5 DOF + gripper).

    Reads joint positions via sync_read, scales normalized values to
    MuJoCo radian targets, and outputs:
      {shoulder_pan.pos, shoulder_lift.pos, elbow_flex.pos,
       wrist_flex.pos, wrist_roll.pos, gripper.pos}

    Output is always in MuJoCo radians, ready for direct use by
    MuJoCo robot's send_action (position mode).
    """

    # NOTE: `name` only controls the calibration directory
    # (HF_LEROBOT_CALIBRATION/teleoperators/<name>/). It is intentionally
    # different from the registered config `type` ("so101_leader_arm"): the
    # type must avoid clashing with LeRobot's built-in "so101_leader", while the
    # calibration dir is safe as "so101_leader" (built-in leader uses "so_leader").
    name = "so101_leader"
    config_class = SO101LeaderTeleopConfig

    def __init__(self, config: SO101LeaderTeleopConfig):
        super().__init__(config)

    def get_action(self) -> dict[str, float]:
        """Read leader arm positions and scale to MuJoCo radians."""
        raw = super().get_action()
        scaled = {}
        for key, val in raw.items():
            joint = key.removesuffix(".pos")
            lo, hi = _LEADER_RANGE_GRIPPER if joint == "gripper" else _LEADER_RANGE_ARM
            mj_lo, mj_hi = _MUJOCO_JOINT_RANGE[joint]
            t = (val - lo) / (hi - lo)
            scaled[key] = mj_lo + t * (mj_hi - mj_lo)
        if _GRIPPER_DEBUG:
            g_raw = raw.get("gripper.pos")
            g_scaled = scaled.get("gripper.pos")
            if g_raw is not None:
                print(
                    f"[gripper] raw_norm={g_raw:8.3f}  scaled_rad={g_scaled:8.4f}",
                    flush=True,
                )
        return scaled
