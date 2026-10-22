from __future__ import annotations

from types import SimpleNamespace

import mujoco as mj
import numpy as np
import pytest
import torch
from lerobot.processor import TransitionKey

from simstudio.common.so101_eef_kinematics import SO101EEFAdapter
from simstudio.policies.lingbot_eef import LingBotEEFToJointProcessorStep


@pytest.fixture(scope="module")
def adapter() -> SO101EEFAdapter:
    return SO101EEFAdapter()


def test_joint_target_becomes_seven_dimensional_eef_command(
    adapter: SO101EEFAdapter,
) -> None:
    current = np.array([0.0, -0.28, 0.62, 1.2, 0.0, 0.85])
    target = np.array([0.01, -0.27, 0.60, 1.19, 0.02, -0.1])

    command = adapter.joint_target_to_eef_command(current, target)

    assert command.shape == (7,)
    assert np.isfinite(command).all()
    assert np.max(np.abs(command[:6])) <= 1.0
    assert command[6] == 1.0


@pytest.mark.parametrize("seed", range(5))
def test_fk_eef_ik_roundtrip_recovers_pose(
    adapter: SO101EEFAdapter, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    current = np.array([0.0, -0.28, 0.62, 1.2, 0.0, 0.85])
    target = current.copy()
    target[:5] += rng.normal(0.0, 0.015, size=5)
    target[5] = -0.1
    command = adapter.joint_target_to_eef_command(current, target, clip=False)

    reconstructed, info = adapter.eef_command_to_joint_target(current, command)
    expected_position, expected_rotation = adapter.forward(target)
    actual_position, actual_rotation = adapter.forward(reconstructed)

    assert info.converged
    assert np.linalg.norm(expected_position - actual_position) < 3e-4
    rotation_cos = np.clip(
        (np.trace(expected_rotation @ actual_rotation.T) - 1.0) / 2.0,
        -1.0,
        1.0,
    )
    assert np.arccos(rotation_cos) < 4e-3
    assert reconstructed[5] == pytest.approx(adapter.gripper_closed)


def test_gripper_sign_maps_to_open_and_closed(adapter: SO101EEFAdapter) -> None:
    current = np.array([0.0, -0.28, 0.62, 1.2, 0.0, 0.85])
    command = np.zeros(7)

    command[6] = -1.0
    opened, _ = adapter.eef_command_to_joint_target(current, command)
    command[6] = 1.0
    closed, _ = adapter.eef_command_to_joint_target(current, command)

    assert opened[5] == pytest.approx(adapter.gripper_open)
    assert closed[5] == pytest.approx(adapter.gripper_closed)


def test_processor_converts_lingbot_tensor_to_six_joints(
    adapter: SO101EEFAdapter,
) -> None:
    names = (*("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"), "gripper")
    dof_ids = {
        name: int(
            adapter.model.jnt_dofadr[
                mj.mj_name2id(adapter.model, mj.mjtObj.mjOBJ_JOINT, name)
            ]
        )
        for name in names
    }
    act_ids = {
        name: mj.mj_name2id(adapter.model, mj.mjtObj.mjOBJ_ACTUATOR, name)
        for name in names
    }
    robot = SimpleNamespace(
        config=SimpleNamespace(
            xml_path="SO101/scenes/simple_pick/scene.xml",
            ee_site_name="wrist_site",
            home_gripper=0.85,
        ),
        model=adapter.model,
        data=adapter.data,
        dof_ids=dof_ids,
        act_ids=act_ids,
        JOINT_NAMES=names,
    )
    current = np.array([0.0, -0.28, 0.62, 1.2, 0.0, 0.85])
    for name, value in zip(names, current, strict=True):
        robot.data.qpos[dof_ids[name]] = value
    mj.mj_forward(robot.model, robot.data)
    step = LingBotEEFToJointProcessorStep(robot)

    output = step({TransitionKey.ACTION: torch.zeros(7)})

    assert output[TransitionKey.ACTION].shape == (6,)
    assert torch.isfinite(output[TransitionKey.ACTION]).all()
