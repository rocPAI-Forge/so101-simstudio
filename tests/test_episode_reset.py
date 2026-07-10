"""Tests for MuJoCo episode auto-reset during recording."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from simstudio.robots.so101_mujoco.configuration_so101_mujoco import SO101MujocoConfig
from simstudio.robots.so101_mujoco.robot_so101_mujoco import SO101MujocoRobot
from simstudio.scripts import record as record_module


class _FakeDataset:
    def __init__(self, num_episodes: int = 0):
        self.num_episodes = num_episodes


def test_maybe_auto_reset_skips_when_dataset_is_none():
    robot = MagicMock()
    record_module._maybe_auto_reset_episode(robot, None)
    robot.reset_episode.assert_not_called()


def _make_robot_mock(reset_mode: str) -> MagicMock:
    robot = MagicMock()
    robot.config.reset_mode = reset_mode
    return robot


def test_maybe_auto_reset_skips_manual_mode():
    robot = _make_robot_mock("manual")
    record_module._maybe_auto_reset_episode(robot, _FakeDataset(2))
    robot.reset_episode.assert_not_called()


def test_maybe_auto_reset_calls_reset_episode():
    robot = _make_robot_mock("auto")
    record_module._maybe_auto_reset_episode(robot, _FakeDataset(2))
    robot.reset_episode.assert_called_once_with(2)


@pytest.mark.parametrize("episode_index", [0, 1, 2])
def test_reset_episode_headless(episode_index: int):
    cfg = SO101MujocoConfig(
        render_window=False,
        camera_names=["front"],
        reset_mode="auto",
    )
    robot = SO101MujocoRobot(cfg)
    robot.connect()
    try:
        robot.reset_episode(episode_index)
        pos = robot.get_block_position()
        assert pos is not None
    finally:
        robot.disconnect()


def test_skip_empty_episode_save_patch():
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = MagicMock()
    dataset.has_pending_frames.return_value = False
    dataset.clear_episode_buffer = MagicMock()

    LeRobotDataset.save_episode(dataset)

    dataset.clear_episode_buffer.assert_called_once()
    dataset.writer.save_episode.assert_not_called()
