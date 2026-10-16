# External VLA inference

SimStudio can evaluate a VLA that is not built into LeRobot without modifying
the pinned `lerobot/` submodule. The integration has three layers:

1. A model-neutral SO-101 contract in
   `simstudio.policies.external_vla`.
2. The separately packaged `lerobot_policy_external_vla` proxy, which lets the
   existing LeRobot rollout and RTC engines call that contract.
3. A model adapter, running either in `.venv-rocm` or in a dependency-isolated
   sidecar.

Training remains in each model's native stack. Consuming a LeRobot v3 dataset
is not sufficient by itself: a deployable checkpoint must also export a
deployment manifest, normalization data, processor/action-decoder settings,
and a pinned model revision.

## Canonical boundary

The request contains named SO-101 data:

- `camera_front`, `camera_top`, and/or `camera_wrist`, RGB uint8;
- six named joint positions in radians;
- optional six joint velocities and end-effector XYZ;
- task text and an observation timestamp.

Every adapter returns an action chunk shaped `[T, 6]` in this fixed order:

```text
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Actions are absolute joint positions in radians. Padded model tensors,
relative actions, normalization, embodiment selection, EEF commands, and IK
must not cross this boundary; the model adapter resolves them.

## Install the LeRobot bridge

`make rocm-sync` installs the bridge automatically. For an existing
environment:

```bash
uv pip install --python .venv-rocm/bin/python --no-deps \
  -e plugins/lerobot_policy_external_vla
```

It is intentionally a separate distribution. LeRobot discovers installed
packages whose names start with `lerobot_policy_`; putting the same module
inside the main SimStudio distribution would not trigger plugin discovery.

## Fake sidecar smoke

Create a local policy bundle:

```bash
.venv-rocm/bin/python -m simstudio.scripts.create_external_vla_bundle \
  --manifest configs/external_vla/fake_manifest.json \
  --output outputs/external_vla/fake \
  --backend sidecar
```

Start the sidecar:

```bash
.venv-rocm/bin/python -m simstudio.policies.external_vla.sidecar \
  --manifest outputs/external_vla/fake/deployment_manifest.json \
  --host 127.0.0.1 --port 10086
```

Run a short closed-loop smoke in another terminal:

```bash
LAB01_POLICY_PATH=outputs/external_vla/fake \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_external_vla.yaml \
LAB01_EVAL_EPISODES=1 LAB01_EPISODE_TIME_S=2 \
  ./labs/lab01_pnp/eval.cmd
```

The fake adapter holds the current joints. It validates transport, plugin
discovery, episode reset, RTC chunk flow, and action safety; it is not expected
to solve pick-and-place.

For an in-process smoke, create the bundle with `--backend inprocess`; no
server is needed. In-process adapters must be importable with dependencies
compatible with `.venv-rocm`.

## Sidecar protocol

The HTTP v1 API is deliberately small and versioned:

- `GET /v1/health`
- `GET /v1/capabilities`
- `POST /v1/reset`
- `POST /v1/predict`

Images are JPEG encoded for the first implementation. The response reports
server preprocessing/model timing, while the client adds round-trip time.
Shared memory should only be added after measurements show JPEG/loopback is a
bottleneck.

The local `SafeBackend` checks capabilities, observation fields, response
timestamps, horizon, finite values, optional joint limits, and stale
responses. A timeout or disconnected server cannot synthesize a replacement
action. Once the RTC queue drains, the robot receives no new command.

## Deployment manifest

Examples live in `configs/external_vla/`. A manifest pins:

- adapter import target;
- model repository/path and exact revision;
- required canonical cameras/state fields;
- control rate, prediction horizon, and execution horizon;
- fixed action semantics and units;
- model-specific mapping and normalization;
- canonical joint limits enforced locally before a chunk enters RTC.

`create_external_vla_bundle` copies the manifest and writes the LeRobot
`config.json` and processor JSON files expected by rollout. The proxy itself
has no trainable model weights and overrides `from_pretrained`; the actual
weights remain owned by the adapter or sidecar.

## Reference adapters

### MiniCPM-RobotManip

`MiniCPMRobotManipAdapter` implements the published Transformers inference
path, but it intentionally refuses to start until the manifest supplies:

- a valid `embodiment_id`;
- six unique state channel indices in the 80-D state;
- six unique action channel indices in the 80-D output;
- state/action affine normalization values.

The public release does not document an SO-101 mapping. The checked-in example
contains invalid `-1`/empty placeholders so it can never accidentally drive
the robot. Obtain an authoritative mapping or fine-tune/export one before
closed-loop use.

### Xiaomi XR-1

`XiaomiXR1Adapter` is intended to run in the upstream CUDA/FlashAttention
environment. Its manifest provides two environment-local hooks:

- `runtime_factory`: constructs a wrapper around the upstream XR-1 client;
- `ik_solver`: converts the selected arm's EEF action components into
  `[T, 6]` absolute SO-101 joint targets.

The adapter maps canonical cameras to XR-1's `ego`, `wrist_left`, and
`wrist_right` roles, packs one SO-101 arm into the selected left/right slot,
and requires the native runtime to own its training normalization. The
example import targets are placeholders and must be implemented in the XR-1
environment.

## Admission gates for a real model

Use the following order and stop at the first failure:

1. Load model and validate manifest/capabilities.
2. Replay one recorded observation through the adapter.
3. Run open-loop dataset metrics and inspect action units/ranges.
4. Pass the same golden observation through in-process and sidecar paths.
5. Run a fixed-start closed-loop smoke.
6. Run randomized closed-loop evaluation.

For 20 Hz control, the steady-state RTC capacity must satisfy:

```text
chunk_hz * execution_horizon >= 20
```

Report model and round-trip latency separately. A WAN result that misses this
gate is an offline/open-loop result, not a comparable closed-loop evaluation.
