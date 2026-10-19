# RAX drawer prototype

This lab adds a task-level environment and RAX-style robot backend for opening the existing SO-101 `smart_drawer` MuJoCo scene. It leaves the LeRobot recording and policy-training workflows unchanged.

## Task and success contract

- Goal: grasp the lower drawer handle and pull toward the SO-101.
- Drawer joint: `drawer_slide`, physical range 0–0.08 m. The green button is decorative; there is no lock behavior.
- Success: open at least 0.06 m farther than the drawer's settled position at episode reset. Valid initial openings are 0–0.02 m, leaving enough travel.
- Seed is metadata because this scene is deterministic; `drawer_position_m` selects the initial drawer opening.
- Each velocity component is bounded by the configured speed limits; a tool call lasts no longer than 2 s.

## How the implementation works

### Environment (`environment.py`)

`RaxDrawerEnv` wraps the existing `SO101MujocoRobot` and the scene `SO101/scenes/smart_drawer/scene.xml`.

1. It configures velocity control, the existing `front`, `top`, and `wrist` cameras, and the actual `gripperframe` site for end-effector control and pose reporting.
2. `reset()` sends the arm to its home pose, resets the scene block, puts the drawer at the requested opening, clears its velocity, advances the simulation briefly to settle, and then records that settled position as the episode baseline.
3. `observe()` returns the drawer sensor value, baseline-relative opening, success threshold, joint positions/velocities, end-effector position, and optionally three RGB camera frames.
4. `step()` clamps world-frame velocity and gripper/wrist actions, advances a bounded duration through the existing robot controller, and returns a fresh observation plus termination/success state.
5. `check_success()` compares the drawer's sensor reading with the reset baseline. It does not rely on the action requested or the seed.

The supported action fields are `vx`, `vy`, `vz` in m/s, `wrist_flex_rate` and `yaw_rate` in rad/s, and `gripper_delta` in the SO101 velocity controller's rate convention. Positive Y pulls the drawer toward the robot; negative `gripper_delta` closes the gripper.

### RAX backend (`backend.py`)

`RaxDrawerBackend` implements the `real/franka` robot-service shape expected by the RAX runtime: `tools()`, `call()`, `status()`, and `close()`. It exposes four calls:

- `drawer_reset`: start a fresh episode.
- `drawer_observe`: return state and camera images.
- `drawer_step`: apply one bounded action and return the updated state.
- `drawer_check_success`: report whether the episode goal is met.

RGB arrays are encoded as PNG MCP image blocks so an image-capable agent can inspect the scene. The backend adapts the existing simulator; it does not alter the MuJoCo scene format or train a policy.

### Runtime and demonstration

The positive-control sequence in `smoke.py` and `live_gui.py` is a deterministic scripted demonstration: wrist adjustment, approach, close, then short positive-Y pulls with feedback checks. It validates reachability and the task contract. It is **not** an autonomous RAX/LLM policy rollout.

For RAX integration, register the backend in a runtime branch that contains `services/backends.py`:

```python
BACKENDS["so101_drawer"] = "labs.rax_drawer.backend:RaxDrawerBackend"
```

Put both the RAX runtime root and this repository root on `PYTHONPATH`, and configure the service to use this repository's `.venv-rocm/bin/python`. Start a session with `robot="so101_drawer"` and this directory's `TASK.md` as its task description. Optional backend settings include `scene_path`, `camera_names`, `success_position_m`, `episode_limit_s`, and action limits. A detached `origin/real/franka` worktree was used to validate the RAX frontend call path; registry wiring is still a runtime-side integration step and is not part of this repository's commits.

## Setup

On the target Linux host, use the repository's normal setup (including submodules and `make rocm-sync`). The tested environment on halo used Python 3.12, Torch `2.14.1+rocm7.2`, and MuJoCo `3.16.0`.

From the repository root on halo:

```bash
cd ~/Repo/RAX/so101-simstudio
.venv-rocm/bin/python -m labs.rax_drawer.smoke
```

The smoke checks scene loading, the RAX tool contract, drawer sensor reads, reset from a displaced/moving drawer, episode-relative scoring, action bounds, PNG packing, and a no-op action.

### Reproduce the checks

```bash
# Render and package front/top/wrist camera frames (headless EGL)
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.smoke --with-cameras

# Positive-control pull at 0, 1, and 2 cm initial drawer openings
.venv-rocm/bin/python -m labs.rax_drawer.smoke --pull-demo

# Interactive three-camera GUI replay; run from the active desktop session
DISPLAY=:0 .venv-rocm/bin/python -m labs.rax_drawer.live_gui
```

`live_gui` opens the Lab01 task views (`top`, `front`, and `wrist`) in the SO101 MuJoCo camera grid, omits this scene's optional fourth `overview` pane, runs the same bounded positive-control pull, then leaves the completed state visible until the window is closed. If started from SSH rather than a desktop terminal, use the active session's `DISPLAY` and `XAUTHORITY` values so the window is shown on the logged-in desktop. The camera smoke uses EGL; the live window uses GLFW.

## Verified result

On halo, the feedback-driven pull smoke reached the 6 cm success threshold from all three tested starts:

| Initial opening | Final additional opening | Pull chunks |
| ---: | ---: | ---: |
| 0 cm | 6.03 cm | 6 |
| 1 cm | 6.10 cm | 6 |
| 2 cm | 6.02 cm | 8 |

The three-camera live GUI replay also completed successfully at about 6.05 cm opening. These figures validate this simulator scene and scripted positive control; they do not measure an autonomous policy's success rate.

## Files

- `environment.py`: task environment and episode contract.
- `backend.py`: RAX robot-service adapter and PNG camera responses.
- `smoke.py`: contract checks and optional feedback-driven positive control.
- `live_gui.py`: Lab01-style three-camera live demonstration.
- `TASK.md`: concise task instructions suitable for an agent session.