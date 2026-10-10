"""Hand a near, still-open grasp to the scripted expert during eval.

The policy keeps approaching. Once the fingertip is within ``LAB01_TAKEOVER_DIST``
(default 8 cm) and the gripper is still open, the expert aligns the wrist, creeps
in, requires a two-jaw contact, lifts, and places. This is a hybrid controller:
quote it next to the success rate. It does not change training.
"""

from __future__ import annotations

import importlib
import logging
import os
import sys
from pathlib import Path

import mujoco as mj
import numpy as np

from simstudio.common.eval_success import CONTAINER_XY

logger = logging.getLogger(__name__)


class ExpertTakeover:
    def __init__(self) -> None:
        self.dist_m = float(os.environ.get("LAB01_TAKEOVER_DIST", "0.08"))
        self.open_q = float(os.environ.get("LAB01_TAKEOVER_OPEN_Q", "0.45"))
        self._used = False
        self.fired = False
        self.lifted = False
        self.success = False
        lab = Path(__file__).resolve().parents[3] / "labs" / "lab01_pnp"
        if str(lab) not in sys.path:
            sys.path.insert(0, str(lab))
        self._yaw = importlib.import_module("auto_collect_yaw")

    def consider(self, robot, tracker) -> bool:
        """Run the expert once, then tell the eval loop to stop the policy."""
        if self._used or tracker.stages["lift_2cm"]:
            return False
        block = robot.get_block_position()
        if block is None or block[2] >= tracker.initial_z + 0.02:
            return False
        grip_id = mj.mj_name2id(robot.model, mj.mjtObj.mjOBJ_JOINT, "gripper")
        if grip_id < 0:
            return False
        qadr = int(robot.model.jnt_qposadr[grip_id])
        if float(robot.data.qpos[qadr]) < self.open_q:
            return False
        site = robot.data.site_xpos[tracker.site_id]
        dist = float(np.linalg.norm(np.asarray(site) - np.asarray(block)))
        if dist > self.dist_m:
            return False
        self._used = True
        self.fired = True
        logger.info(
            "Expert takeover at tick=%d dist=%.3f cube=%s",
            tracker.tick,
            dist,
            tuple(round(float(v), 3) for v in block),
        )
        self._run(robot, tracker)
        return True

    def _run(self, robot, tracker) -> None:
        yaw = self._yaw
        yaw.STEP_SLEEP_S = 0.0
        previous = yaw._frame_hook

        def _hook(_obs, sent) -> None:
            tracker.update(sent)

        yaw._frame_hook = _hook
        try:
            attempt = self._lift(robot, yaw, "takeover-")
            if not attempt.get("lifted"):
                logger.info("Expert takeover retry from a fresh standoff")
                cx, cy, _cz, cube_yaw = yaw._block_pose(robot)
                yaw._prepare_regrasp(robot, np.array([cx, cy], dtype=float), cube_yaw)
                attempt = self._lift(robot, yaw, "takeover-retry-")
            self.lifted = bool(attempt.get("lifted"))
            wrist_target = attempt.get("wrist_target")
            if self.lifted and wrist_target is not None:
                self._place(robot, yaw, np.asarray(wrist_target, dtype=float))
            block = robot.get_block_position()
            from simstudio.common.eval_success import check_pick_success

            self.success = bool(check_pick_success(block))
            logger.info(
                "Expert takeover done lifted=%s success=%s block=%s",
                self.lifted,
                self.success,
                block,
            )
        finally:
            yaw._frame_hook = previous

    def _lift(self, robot, yaw, prefix: str) -> dict:
        cx, cy, cz, cube_yaw = yaw._block_pose(robot)
        return yaw._attempt_lift(
            robot,
            None,
            cx=cx,
            cy=cy,
            cz=cz,
            cube_yaw_deg=cube_yaw,
            spawn=(cx, cy, cz),
            close_dx=0.0,
            close_dy=0.0,
            close_droll=0.0,
            gripper_closed=None,
            phase_prefix=prefix,
            contact_reach_m=0.012,
            slip_max_m=0.012,
        )

    def _place(self, robot, yaw, wrist_target: np.ndarray) -> None:
        yaw.servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], yaw.LIFT_Z]),
            gripper=yaw.GRIPPER_CLOSED,
            steps=140,
            phase="takeover-to-box",
            lock_wrist=True,
            wrist_target=wrist_target,
            kp=1.6,
            dq_lim=0.35,
            prevent_overshoot=False,
            max_step=1.0,
        )
        yaw.hold(robot, yaw.GRIPPER_CLOSED, steps=10)
        yaw.servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], yaw.PLACE_Z]),
            gripper=yaw.GRIPPER_CLOSED,
            steps=70,
            phase="takeover-lower",
            lock_wrist=True,
            wrist_target=wrist_target,
            prevent_overshoot=False,
            max_step=1.0,
        )
        yaw.set_grip(robot, yaw.GRIPPER_OPEN, steps=30)
        yaw.servo_xyz(
            robot,
            np.array([CONTAINER_XY[0], CONTAINER_XY[1], yaw.LIFT_Z]),
            gripper=yaw.GRIPPER_OPEN,
            steps=45,
            phase="takeover-retreat",
            lock_wrist=False,
            prevent_overshoot=False,
            max_step=1.0,
        )
