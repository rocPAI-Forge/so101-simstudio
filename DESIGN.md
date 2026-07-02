# Design Document

## Goals

Build a clean, maintainable SO-101 teleoperation and dataset-collection project that:

1. Uses latest HuggingFace LeRobot as a submodule **without modifying it**.
2. Runs SO-101 in MuJoCo simulation for keyboard teleop (MVP).
3. Is architected to support Joy-Con, gamepad, and real SO-101 leader-arm teleop.
4. Supports ROCm natively via `uv`.
5. Uses Python 3.12+.

## Non-Goals (for MVP)

- Training pipelines (planned but not in first iteration).
- Real robot hardware drivers (interface stubbed, implementation later).
- Joy-Con / gamepad implementation (interface stubbed, implementation later).

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LeRobot scripts (lerobot_record, lerobot_replay, ...)  │
│  No modifications allowed                                │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│  Third-party plugin discovery                           │
│  lerobot_robot_* / lerobot_teleoperator_*               │
└─────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                           ▼
┌───────────────┐                       ┌─────────────────┐
│  Robot Layer  │                       │  Teleop Layer   │
│               │                       │                 │
│ so101_mujoco  │                       │ so101_keyboard  │
│ so101_leader  │                       │ so101_joycon    │
│ (planned)     │                       │ so101_gamepad   │
│               │                       │ (planned)       │
└───────┬───────┘                       └────────┬────────┘
        │                                        │
        │         ┌──────────────────┐           │
        └────────►│  Action Mapping  │◄──────────┘
                  │                  │
                  │ Convert teleop   │
                  │ output to robot- │
                  │ native action    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ robot.send_action│
                  │                 │
                  └─────────────────┘
```

## Action Semantics

All teleoperators produce a **normalized velocity command**:

```python
{
    "vx": float,              # m/s, end-effector X velocity
    "vy": float,              # m/s, end-effector Y velocity
    "vz": float,              # m/s, end-effector Z velocity
    "wrist_flex_rate": float, # rad/s, wrist flex joint velocity
    "yaw_rate": float,        # rad/s, wrist roll joint velocity
    "gripper_delta": float,   # rad/s, gripper opening rate
}
```

The robot layer converts this into its native action format:

- **MuJoCo robot**: velocities are fed into a Jacobian-based controller that outputs target joint positions.
- **Real leader robot**: velocities are integrated to target joint positions and tracked by the physical arm.

This decoupling means any teleoperator can drive any robot without code changes.

## Package Layout

```
so101-mujoco-teleop/
├── lerobot/                          # HF LeRobot submodule
├── src/
│   └── so101_mujoco_teleop/
│       ├── __init__.py
│       ├── common/
│       │   ├── constants.py          # Joint names, limits, defaults
│       │   └── action_mapping.py     # Teleop output → robot action
│       ├── robots/
│       │   ├── so101_mujoco/         # MuJoCo simulation
│       │   └── so101_leader/         # Real leader arm (stub)
│       ├── teleoperators/
│       │   ├── so101_keyboard/       # Keyboard teleop
│       │   └── so101_joycon/         # Joy-Con teleop (stub)
│       └── scripts/
│           ├── record.py             # Optional LeRobot wrapper
│           └── train.py              # Custom training (later)
├── SO101/                            # MuJoCo assets
├── configs/
│   ├── so101_mujoco_keyboard.yaml
│   └── so101_mujoco_joycon.yaml
├── scripts/
│   └── setup-rocm.sh
├── pyproject.toml
├── README.md
└── AGENTS.md
```

## LeRobot Plugin Mechanism

Latest LeRobot discovers third-party packages by prefix:

- `lerobot_robot_*` → registered as robot implementations.
- `lerobot_teleoperator_*` → registered as teleoperator implementations.

We expose our packages under these names via `pyproject.toml` optional or namespace packages, or by shipping subpackages named `lerobot_robot_so101_mujoco`, etc.

For simplicity, this project may use a single package with namespace subpackages:

```
lerobot_robot_so101_mujoco/
lerobot_robot_so101_leader/
lerobot_teleoperator_so101_keyboard/
lerobot_teleoperator_so101_joycon/
```

All pointing back to `src/so101_mujoco_teleop` modules.

## Migration from robopicker

Code to port/adapt:

1. `robopicker/lerobot/src/lerobot/robots/so101_mujoco/robot_so101_mujoco.py`
   - Adapt to latest LeRobot `Robot` base class.
   - Return `RobotAction`/`RobotObservation` type aliases.
   - Ensure `disconnect()` is safe under `__del__`.
   - Fix GLFW window hints (default hints, no FOCUSED=FALSE).

2. `robopicker/lerobot/src/lerobot/robots/so101_mujoco/configuration_so101_mujoco.py`
   - Keep `@register_subclass("so101_mujoco")` pattern.
   - Verify draccus/dataclass compatibility with latest LeRobot.

3. `robopicker/lerobot/src/lerobot/teleoperators/keyboard/teleop_so101_keyboard.py`
   - Replace `lerobot.utils.keyboard_event_manager` with `lerobot.utils.keyboard_input`.
   - Output normalized velocity dict.

4. `robopicker/src/robopicker/scripts/train.py`
   - Port later for training with train/eval splits.

5. `robopicker/SO101/`
   - Copy MuJoCo assets and scene XML.

6. `robopicker/configs/so101_mujoco_record.yaml`
   - Adapt to latest `DatasetRecordConfig` location and fields.

## Testing Strategy

1. **Unit / smoke tests**: verify plugin registration, config loading, action mapping.
2. **Integration test**: open MuJoCo scene preview.
3. **End-to-end test**: record one short episode with keyboard teleop and load it back.
4. **Window visibility test**: on Ubuntu 24.04, verify recording window appears.

## Risks

1. **LeRobot API churn**: latest LeRobot may still change; pin submodule to a known good commit after validation.
2. **ROCm + Python 3.12**: torch ROCm wheels availability must be verified.
3. **Joy-Con Linux support**: depends on `hid-nintendo` / `joycond` drivers; may require manual setup.
4. **Submodule size**: HF LeRobot is large; shallow clones recommended for CI.

## Next Steps

1. Implement `so101_mujoco` robot package.
2. Implement `so101_keyboard` teleop package.
3. Wire both via LeRobot plugin discovery.
4. Add recording config and verify end-to-end recording.
5. Add Joy-Con / leader stubs.
6. Add ROCm setup script and verify.
