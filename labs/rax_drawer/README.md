# RAX drawer prototype

A task-level adapter for online exploration of the existing SO-101 `smart_drawer` MuJoCo scene. It does not change the LeRobot recording or policy-training workflows.

## Task contract

- Goal: grasp the lower drawer handle and pull the drawer toward the SO-101.
- Joint: `drawer_slide`, range 0–0.08 m.
- Success: the drawer opens at least 0.06 m relative to its episode start. Start positions are limited to states where that travel remains possible. The scene is deterministic; reset seeds are metadata, while `drawer_position_m` selects the initial opening.
- The green unlock button is decorative; this scene has no lock behavior.
- Velocity actions are bounded; each call lasts at most 2 seconds.
- Observations contain joint state, end-effector position, drawer position, and optional RGB frames.

## Code

- `environment.py`: Python API with `reset()`, `observe()`, `step(action, duration_s)`, `check_success()`, and `close()`. Camera frames are NumPy arrays. Velocity control and the reported end-effector pose use the actual `gripperframe` site.
- `backend.py`: implements the RAX `real/franka` robot-service backend contract (`tools()`, `call()`, `status()`, `close()`) with four tools: `drawer_reset`, `drawer_observe`, `drawer_step`, and `drawer_check_success`. It encodes RGB frames as PNG MCP image blocks.

The base RAX clone remains on its index branch. A detached `origin/real/franka` worktree at `~/Repo/RAX/rax-drawer-runtime` was used to validate the real RAX frontend; no RAX source files were changed. To register this backend in a runtime branch that contains `services/backends.py`, add:

```python
BACKENDS["so101_drawer"] = "labs.rax_drawer.backend:RaxDrawerBackend"
```

Add this repository root to `PYTHONPATH` alongside the selected RAX runtime root, and point the service interpreter at `.venv-rocm/bin/python`. For headless camera rendering, set `MUJOCO_GL=egl` (or another renderer verified on the host). A session uses `robot="so101_drawer"` and `TASK.md` as its `task_description`; optional `backend_options` accept `scene_path`, `camera_names`, `success_position_m`, `episode_limit_s`, and action limits. The RAX worktree used for validation is detached and clean; registry wiring remains a separate integration step.

## Setup and smoke

Use the repository's documented environment: initialize submodules, run `make rocm-sync`, then:

```bash
# State/control smoke; no display or GL context required
.venv-rocm/bin/python -m labs.rax_drawer.smoke

# Optional camera-path smoke on halo
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.smoke --with-cameras

# Optional feedback-driven pull across 0, 1, and 2 cm starts
.venv-rocm/bin/python -m labs.rax_drawer.smoke --pull-demo

# Live Lab01-style top/front/wrist camera window + scripted drawer pull
DISPLAY=:0 .venv-rocm/bin/python -m labs.rax_drawer.live_gui
```

The state/control smoke checks scene loading, the RAX tool contract, drawer sensor readings, reset from a displaced/moving drawer, episode-relative success scoring, action bounds, PNG packing, and a no-op action. The optional camera run also renders front/top/wrist frames and packages them as MCP image blocks. The default and camera modes do not attempt a grasp. The optional `--pull-demo` uses short positive-Y pull actions and checks success after each one across 0, 1, and 2 cm starts. All three start states open by at least 6 cm; this positive-control smoke is separate from the agent policy.

The live GUI command opens the three task camera views (`top`, `front`, and `wrist`) in the SO101 MuJoCo camera grid, then replays a bounded positive-control pull. The completed scene stays visible until the window is closed. It is a deterministic control demonstration, not an autonomous RAX/LLM rollout. Run it from the active desktop session; `DISPLAY` should point at that session.
