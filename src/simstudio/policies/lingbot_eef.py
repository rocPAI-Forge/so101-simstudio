"""LingBot-VA EEF action adapter for SO-101 rollout."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from lerobot.processor import EnvTransition, ProcessorStep, TransitionKey

from simstudio.common.so101_eef_kinematics import SO101EEFAdapter

logger = logging.getLogger(__name__)


class LingBotEEFToJointProcessorStep(ProcessorStep):
    """Project unnormalized LIBERO EEF commands to absolute SO-101 joints."""

    def __init__(self, robot: Any) -> None:
        self.robot = robot
        self.adapter = SO101EEFAdapter(
            robot.config.xml_path,
            robot.config.ee_site_name,
        )
        self.total_actions = 0
        self.failed_convergence = 0
        self.max_position_error_m = 0.0
        self.max_orientation_error_rad = 0.0

    def _current_joints(self) -> np.ndarray:
        return np.asarray(
            [self.robot.data.qpos[self.robot.dof_ids[name]] for name in self.robot.JOINT_NAMES],
            dtype=np.float64,
        )

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        action = transition.get(TransitionKey.ACTION)
        if action is None or not isinstance(action, torch.Tensor):
            return transition
        if action.shape[-1] != 7:
            raise ValueError(
                f"LingBot EEF adapter expected action[...,7], got {tuple(action.shape)}"
            )

        original_shape = action.shape
        flat = action.detach().cpu().reshape(-1, 7).numpy()
        current = self._current_joints()
        converted = []
        for command in flat:
            target, info = self.adapter.eef_command_to_joint_target(current, command)
            converted.append(target)
            current = target
            self.total_actions += 1
            if not info.converged:
                self.failed_convergence += 1
            self.max_position_error_m = max(
                self.max_position_error_m, info.position_error_m
            )
            self.max_orientation_error_rad = max(
                self.max_orientation_error_rad, info.orientation_error_rad
            )

        output = torch.as_tensor(
            np.stack(converted),
            dtype=action.dtype,
            device=action.device,
        ).reshape(*original_shape[:-1], 6)
        updated = dict(transition)
        updated[TransitionKey.ACTION] = output
        return updated

    def reset(self) -> None:
        if self.total_actions:
            logger.info(
                "LingBot EEF IK: actions=%d nonconverged=%d max_pos_err=%.4fm "
                "max_ori_err=%.3frad",
                self.total_actions,
                self.failed_convergence,
                self.max_position_error_m,
                self.max_orientation_error_rad,
            )
        self.total_actions = 0
        self.failed_convergence = 0
        self.max_position_error_m = 0.0
        self.max_orientation_error_rad = 0.0

    def transform_features(self, features):
        return features


def install_lingbot_eef_adapter(ctx: Any, robot: Any) -> None:
    """Append EEF→joint conversion after LingBot's quantile unnormalizer."""
    policy_type = str(getattr(ctx.policy.policy.config, "type", "")).lower()
    class_name = type(ctx.policy.policy).__name__.lower()
    if "lingbot" not in policy_type and "lingbot" not in class_name:
        raise TypeError(
            f"LAB01_LINGBOT_EEF_ADAPTER requires LingBot-VA, got {type(ctx.policy.policy)}"
        )
    ctx.policy.postprocessor.steps.append(LingBotEEFToJointProcessorStep(robot))
    logger.info(
        "Installed LingBot EEF→SO-101 joint adapter "
        "(LIBERO scales: translation=0.05m, rotation=0.5rad)"
    )
