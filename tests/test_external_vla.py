"""Contract, safety, and sidecar tests for external VLA inference."""

from __future__ import annotations

import json
import sys
import threading
import time
import types

import numpy as np
import pytest

from simstudio.common.constants import JOINT_NAMES
from simstudio.policies.external_vla import (
    ActionChunk,
    ContractError,
    DeploymentManifest,
    InProcessBackend,
    ObservationFrame,
    SafeBackend,
    SidecarClient,
)
from simstudio.policies.external_vla.adapters.fake import FakeAdapter
from simstudio.policies.external_vla.adapters.minicpm_robotmanip import (
    MiniCPMRobotManipAdapter,
)
from simstudio.policies.external_vla.adapters.xiaomi_xr1 import XiaomiXR1Adapter
from simstudio.policies.external_vla.sidecar import SidecarError, create_server


def _manifest(**overrides) -> DeploymentManifest:
    values = {
        "adapter": "simstudio.policies.external_vla.adapters.fake:FakeAdapter",
        "model_id": "simstudio/fake",
        "model_revision": "test-v1",
        "required_cameras": ("camera_top",),
        "control_fps": 20.0,
        "action_horizon": 4,
        "execution_horizon": 2,
        "adapter_config": {"joint_delta_per_step": [0.01, 0, 0, 0, 0, 0]},
        "joint_limits": {name: (-10.0, 10.0) for name in JOINT_NAMES},
    }
    values.update(overrides)
    return DeploymentManifest(**values)


def _observation(*, timestamp_ns: int | None = None) -> ObservationFrame:
    return ObservationFrame(
        task="Pick up the cube.",
        timestamp_ns=timestamp_ns or time.time_ns(),
        joint_positions={name: index / 10.0 for index, name in enumerate(JOINT_NAMES)},
        images={"camera_top": np.full((16, 24, 3), 127, dtype=np.uint8)},
    )


def test_manifest_round_trip(tmp_path):
    manifest = _manifest()
    path = tmp_path / "deployment_manifest.json"
    manifest.save(path)
    assert DeploymentManifest.load(path) == manifest
    payload = json.loads(path.read_text())
    assert payload["action_semantics"] == "absolute_joint_position"
    assert payload["model_revision"] == "test-v1"


def test_manifest_rejects_noncanonical_actions():
    with pytest.raises(ContractError, match="action_semantics"):
        _manifest(action_semantics="relative_joint_delta").validate()


def test_observation_wire_round_trip():
    source = _observation()
    restored = ObservationFrame.from_wire(source.to_wire(jpeg_quality=95))
    assert restored.task == source.task
    assert restored.timestamp_ns == source.timestamp_ns
    assert restored.joint_positions == source.joint_positions
    assert restored.images["camera_top"].shape == source.images["camera_top"].shape


def test_observation_requires_canonical_joint_order():
    observation = _observation()
    reversed_positions = dict(reversed(observation.joint_positions.items()))
    invalid = ObservationFrame(
        task=observation.task,
        joint_positions=reversed_positions,
        images=observation.images,
    )
    with pytest.raises(ContractError, match="canonical order"):
        invalid.validate()


def test_observation_rejects_missing_required_camera():
    with pytest.raises(ContractError, match="Missing required cameras"):
        _observation().validate(required_cameras=("camera_front",))


def test_action_chunk_rejects_nan_and_stale():
    now = time.time_ns()
    nan_chunk = ActionChunk(
        actions=np.full((2, 6), np.nan, dtype=np.float32),
        source_timestamp_ns=now,
        generated_at_ns=now,
        dt_s=0.05,
    )
    with pytest.raises(ContractError, match="NaN"):
        nan_chunk.validate()

    stale = ActionChunk(
        actions=np.zeros((2, 6), dtype=np.float32),
        source_timestamp_ns=now - 2_000_000_000,
        generated_at_ns=now - 1_000_000_000,
        dt_s=0.05,
    )
    with pytest.raises(ContractError, match="stale"):
        stale.validate(stale_after_s=0.5, now_ns=now)


def test_safe_backend_rejects_joint_limit_violation():
    limits = {name: (-0.001, 0.001) for name in JOINT_NAMES}
    manifest = _manifest(joint_limits=limits)
    backend = SafeBackend(
        InProcessBackend(manifest, FakeAdapter(manifest)),
        manifest,
        stale_after_s=1.0,
    )
    with pytest.raises(ContractError, match="outside"):
        backend.predict_chunk(_observation())


def test_inprocess_and_sidecar_golden_match():
    manifest = _manifest()
    observation = _observation()
    local = SafeBackend(
        InProcessBackend(manifest, FakeAdapter(manifest)),
        manifest,
        stale_after_s=1.0,
    )
    local_chunk = local.predict_chunk(observation)

    server = create_server(manifest, port=0, adapter=FakeAdapter(manifest))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = SidecarClient(f"http://{host}:{port}", timeout_s=1.0)
    remote = SafeBackend(client, manifest, stale_after_s=1.0)
    try:
        assert remote.health()["ok"] is True
        remote.reset_episode("golden")
        remote_chunk = remote.predict_chunk(observation)
    finally:
        remote.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)

    np.testing.assert_allclose(remote_chunk.actions, local_chunk.actions)
    assert remote_chunk.source_timestamp_ns == observation.timestamp_ns
    assert remote_chunk.timings_ms["round_trip"] >= 0


def test_sidecar_timeout_is_not_converted_to_an_action():
    manifest = _manifest()

    class SlowAdapter(FakeAdapter):
        def predict_chunk(self, observation):
            time.sleep(0.05)
            return super().predict_chunk(observation)

    server = create_server(manifest, port=0, adapter=SlowAdapter(manifest))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    client = SidecarClient(f"http://{host}:{port}", timeout_s=0.01)
    try:
        with pytest.raises(SidecarError, match="failed"):
            client.predict_chunk(_observation())
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1.0)


def test_fake_adapter_capacity_meets_control_rate():
    manifest = _manifest()
    backend = InProcessBackend(manifest, FakeAdapter(manifest))
    observation = _observation()
    started = time.perf_counter()
    count = 20
    for _ in range(count):
        backend.predict_chunk(observation)
    chunk_hz = count / (time.perf_counter() - started)
    assert chunk_hz * manifest.execution_horizon >= manifest.control_fps


def test_minicpm_adapter_fails_closed_without_authoritative_mapping():
    manifest = _manifest(
        adapter=("simstudio.policies.external_vla.adapters.minicpm_robotmanip:MiniCPMRobotManipAdapter"),
        model_id="openbmb/MiniCPM-RobotManip",
        model_revision="pinned-test-revision",
        adapter_config={
            "embodiment_id": -1,
            "state_indices": {name: -1 for name in JOINT_NAMES},
            "action_indices": {name: -1 for name in JOINT_NAMES},
        },
        normalization={
            "state_offset": [],
            "state_scale": [],
            "action_offset": [],
            "action_scale": [],
        },
    )
    with pytest.raises(ValueError, match="unique values"):
        MiniCPMRobotManipAdapter(manifest)


def test_xr1_adapter_runs_runtime_and_ik_hooks(monkeypatch):
    module = types.ModuleType("test_xr1_bridge")

    class Runtime:
        def predict(self, *, images, instruction, robot_state):
            assert set(images) == {"ego", "wrist_left", "wrist_right"}
            assert instruction == "Pick up the cube."
            assert len(robot_state["right_arm_joint"]) == 6
            return {"action_components": {"right_ee": np.zeros((4, 7))}}

    def make_runtime(**kwargs):
        assert kwargs["model_id"] == "XiaomiRobotics/Xiaomi-Robotics-1-5B"
        return Runtime()

    def solve_so101_chunk(*, current_joints, horizon, **kwargs):
        del kwargs
        return np.repeat(current_joints[None, :], horizon, axis=0)

    module.make_runtime = make_runtime
    module.solve_so101_chunk = solve_so101_chunk
    monkeypatch.setitem(sys.modules, module.__name__, module)
    manifest = _manifest(
        adapter=("simstudio.policies.external_vla.adapters.xiaomi_xr1:XiaomiXR1Adapter"),
        model_id="XiaomiRobotics/Xiaomi-Robotics-1-5B",
        model_revision="pinned-test-revision",
        required_cameras=("camera_front", "camera_top", "camera_wrist"),
        required_state_fields=("joint_positions", "end_effector_xyz"),
        adapter_config={
            "runtime_factory": "test_xr1_bridge:make_runtime",
            "ik_solver": "test_xr1_bridge:solve_so101_chunk",
            "native_runtime_handles_normalization": True,
            "arm_slot": "right",
            "action_component_key": "right_ee",
            "camera_mapping": {
                "camera_front": "ego",
                "camera_top": "wrist_left",
                "camera_wrist": "wrist_right",
            },
        },
    )
    observation = ObservationFrame(
        task="Pick up the cube.",
        joint_positions={name: index / 10.0 for index, name in enumerate(JOINT_NAMES)},
        end_effector_xyz=(0.1, 0.2, 0.3),
        images={name: np.zeros((8, 8, 3), dtype=np.uint8) for name in manifest.required_cameras},
    )
    chunk = XiaomiXR1Adapter(manifest).predict_chunk(observation)
    assert chunk.actions.shape == (4, 6)
    np.testing.assert_allclose(chunk.actions[0], list(observation.joint_positions.values()))
