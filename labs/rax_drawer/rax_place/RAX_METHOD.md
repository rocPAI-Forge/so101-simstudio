# How RAX was implemented for SO-101 drawer place

This document explains the **Robot Action eXecution (RAX)** loop used in
`labs/rax_drawer/rax_place/`: principles, module map, tool API, perception /
control design, and how the frozen **15 / 20** place program was produced.
It complements the chronological log in [`PROCESS.md`](PROCESS.md) and the
protocol in [`PROTOCOL.md`](PROTOCOL.md).

Reported score: freeze 3 (`session/CONTROLLER_FROZEN_3`) → **15 / 20** on
`manifests/test.json`. Replay: `./labs/rax_drawer/rax_place/eval_locked.cmd`.

Demo GIF (top RGB + top depth + side RGB, locked test scene 1):
[`../media/open_place_close_rgbd.gif`](../media/open_place_close_rgbd.gif).

---

## 1. What “RAX-style” means here

RAX (as practiced in this lab) is **not** “train a VLA.” It is:

1. A **coding agent** writes and revises a controller program.
2. The agent may interact with the robot **only through a tool API** that
   hides privileged simulator state.
3. Development uses an **explore** scene list; a **held-out test** list is
   scored **once** after the program is frozen.
4. The deliverable is a **compiled, deterministic controller** (Python), not a
   chat transcript.

On this host the official Claude Code / RAX harness credentials were not
available. Cursor filled the agent role and called
`labs.rax_drawer.rax_place.tools.Session` directly. The same four tools mirror
`labs.rax_drawer.backend.RaxDrawerBackend` (`drawer_*`), which is the shape a
real RAX robot service would expose.

```
┌─────────────────────┐     reset / observe / step / check_success
│  Coding agent       │ ──────────────────────────────────────────►
│  (Cursor / Claude)  │                                            │
└─────────────────────┘                                            │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  Session (tools.py)  — strips layout, drawer joint, cube/handle poses,   │
│                        contacts; audits every call to logs/mcp*.jsonl    │
└──────────────────────────────────────────────────────────────────────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RaxDrawerBackend (backend.py)  — validates tool args, packs PNG images  │
└──────────────────────────────────────────────────────────────────────────┘
                                                                   │
                                                                   ▼
┌──────────────────────────────────────────────────────────────────────────┐
│  RaxDrawerEnv (environment.py)  — MuJoCo smart_drawer + SO101 arm        │
│  Evaluator-only: drawer joint, contacts, cube AABB, layout               │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Principles (non-negotiable)

| Rule | Why |
|------|-----|
| Tool-only control | Matches the real-robot contract: no MuJoCo body IDs, no drawer joint for control. |
| Explore vs test split | Prevents overfitting to the 20 locked layouts. |
| Freeze before locked score | Locked numbers are not a tuning signal. |
| Cameras + joints only | Handle / cube pose must be **perceived**, not looked up. |
| Compile a program | Experience becomes reusable code (`code/open_place_close/`), not one-off tool traces. |

**Privileged (evaluator only, never returned by Session):**

- Cabinet layout `(x_m, y_m, yaw_deg)`
- `drawer_slide` position / velocity
- Cube world pose, handle geom poses
- Contact flags used for success (`pull_contact`, `close_contact`)
- `cube_in_drawer` AABB check (used only inside `check_success`)

**Public (agent / controller):**

- Joint positions and velocities (`observation.state`)
- Camera calibration for `top`, `side`, `wrist`
- RGB frames for those cameras
- Overhead metric depth (`depth_images["top"]`, millimeters)
- `terminated` / `success` after a step that ends the episode
- Boolean from `check_success()`

---

## 3. Module map

### 3.1 Lab root (`labs/rax_drawer/`)

| Path | Role |
|------|------|
| `environment.py` | `DrawerTaskConfig`, `RaxDrawerEnv`: reset/observe/step, depth renderer, success scoring including optional `require_cube_placed`. |
| `backend.py` | `RaxDrawerBackend`: RAX robot-service adapter; `tools()` / `call()`; PNG MCP packing for RGB and 16-bit depth. |
| `TASK.md` | Short task text for an agent session. |
| `open_close.py` | Earlier non-privileged **open/close only** program (reported **16 / 20**). Direct env calls, not Session. |
| `place_close.py` | Earlier place variant on the raw env. **Not** the reported 15/20 program; contamination risk if imported. |
| `manifests/explore.json` | Development layouts (50 scenes). |
| `manifests/test.json` | Locked held-out layouts (20 scenes). |
| `smoke.py` / `live_gui.py` | Contract checks and positive-control demos (not the RAX controller). |

### 3.2 RAX place session (`labs/rax_drawer/rax_place/`)

| Path | Role |
|------|------|
| `tools.py` | `Session`: explore/test layout cycling, strip privileged keys, audit log. |
| `AGENTS.md` | Instructions for the coding agent. |
| `PROTOCOL.md` | Explore → compile → freeze → one locked test. |
| `TASK.md` | Place-task brief. |
| `PROCESS.md` | Chronological experiment log and scores. |
| `run_explore.py` | Unattended explore self-tests. |
| `run_test.py` / `eval_locked.cmd` | Locked evaluation after freeze mark exists. |
| `render_place_gif.py` | Offscreen RGB-D + side GIF of one episode. |
| `session/CONTROLLER_FROZEN_*` | Freeze timestamps; round 3 is the reported program. |
| `logs/locked_test_3.json` | Frozen 15/20 result payload. |
| `code/open_place_close/` | Compiled controller package (see below). |
| `memory/NOTES.md` | Agent working notes during explore. |

### 3.3 Compiled controller package

```
code/open_place_close/
  controller.py   # main(ctx): open → pick/place → close
  perceive.py     # lower handle + green cube from top RGB-D
  kinematics.py   # FK: end-effector from joint observation
  ik.py           # Jacobian position servo (joint .pos actions)
  motion.py       # velocity helpers: approach / move / slide
```

Entry point used by explore and locked eval:

```python
from labs.rax_drawer.rax_place.code.open_place_close.controller import main
from labs.rax_drawer.rax_place.tools import Session

with Session(split="test") as ctx:
    ok = main(ctx)   # calls reset once, then the full skill
```

---

## 4. Tool API (what the agent may call)

### 4.1 `Session.reset(*, drawer_position_m=0.0, include_cameras=False) -> dict`

- Privately applies the next explore/test layout (never returned).
- Homes the arm, settles the drawer, records the episode baseline opening.
- Returns a public observation (cameras optional).

### 4.2 `Session.observe(*, include_cameras=True) -> dict`

Typical keys used by the controller:

- `observation.state` — joint `.pos` / `.vel` (and related SO-101 fields)
- `cameras` — `{"top"|"side"|"wrist": HxWx3 uint8}`
- `camera_calibrations` — intrinsics / extrinsics for unprojection
- `depth_images` — `{"top": HxW uint16}` depth in **millimeters**

### 4.3 `Session.step(action, *, duration_s=0.25, include_cameras=False) -> dict`

**Velocity mode** (drawer open/close, coarse slides):

- `vx`, `vy`, `vz` (m/s, world frame)
- `wrist_flex_rate`, `yaw_rate` (rad/s)
- `gripper_delta` (close negative)

**Position mode** (pick IK): joint keys ending in `.pos` (e.g.
`shoulder_pan.pos`, …). The env auto-detects format from key names.
`duration_s` is clamped by the backend (≤ 2 s per call).

Returns observation plus `terminated` / `success` when the episode ends.

### 4.4 `Session.check_success() -> {"success": bool}`

Evaluator-only scoring exposed as a boolean. Place mode requires:

1. Handle contact while opening ≥ 6 cm beyond the episode baseline.
2. Handle contact on return to within 5 mm of that baseline.
3. Green cube resting in the drawer tray (AABB), not held in the gripper.

### 4.5 Backend packing (`backend.py`)

When used as a remote RAX service, RGB becomes PNG MCP image blocks;
depth becomes 16-bit PNG. `Session` keeps numpy arrays in-process for speed
and still strips privileged fields the same way.

---

## 5. Controller design (freeze 3)

### 5.1 Perception (`perceive.py`)

From overhead RGB + depth + calibration:

1. Convert depth mm → meters; unproject valid pixels to world coordinates.
2. **Lower handle:** dark bar in a workspace ROI; fit a line; estimate horizontal
   pull direction toward the fixed arm mount (from kinematics, not layout).
3. **Green cube:** HSV / green mask in the same cloud; take a robust centroid.

No MuJoCo geom IDs. Failures raise `RuntimeError` and abort that episode.

### 5.2 Open drawer (`controller.open_drawer`)

1. Perceive handle + pull axis.
2. Velocity approach with wrist pitch; close gripper with small clearance retries.
3. Confirm pull by **re-perceiving** handle motion or pinch motion along `pull`
   (not by reading `drawer_slide`).
4. Pull ~7.5 cm along `pull` with squeeze; record `grasp_xy` for later close.
5. Release and clear away from the tray so the cube is visible.

### 5.3 Pick and place (`controller.pick_and_place`)

Critical lesson: after opening, **do not** `go_home` with a straight joint path —
the arm sweeps and can shove the open drawer shut. Freeze 3 uses **fold-then-face**:
tuck, then face the cube with Jacobian servos.

1. Perceive cube; open gripper.
2. Jacobian IK (`ik.servo_to`) in joint `.pos` mode to approach → grasp height → close.
3. Lift; move to a drop pose **5.5 cm behind** the perceived handle along `pull`
   (`PLACE_BEHIND_M = 0.055`) at tray height; open gripper.
4. Stay in position mode while transporting — switching back to velocity mid-carry
   dropped the cube in earlier rounds.

### 5.4 Close drawer (`controller.close_drawer`)

Closing after place is harder than open/close-only: the tray is occupied and the
handle may be partially occluded.

1. Pitch away on the robot side of the tray.
2. **Wrist-locked** servo onto the recorded pull axis / handle estimate.
3. Squeeze and push back toward the recorded open grasp point until the evaluator
   accepts return + cube-in-tray.

### 5.5 Constants that matter

| Constant | Value | Meaning |
|----------|-------|---------|
| `PULL_DISTANCE_M` | 0.075 | Open stroke (success needs ≥ 0.06) |
| `PLACE_BEHIND_M` | 0.055 | Drop inside tray, not onto the handle |
| `GRIP_CLOSE` | ≤ −0.05 | Empty-grasp filter for “am I holding?” |
| Episode limit | 180 s | Place horizon is long |

---

## 6. Explore → compile → freeze → locked test

```
AGENTS.md / PROTOCOL.md
        │
        ▼
┌───────────────────┐
│ Explore (Session  │  write short tool traces, notes in memory/
│  split=explore)   │
└─────────┬─────────┘
          │ compile
          ▼
┌───────────────────┐
│ code/open_place_  │ ──── run_explore.py self-test on explore scenes
│ close/controller  │
└─────────┬─────────┘
          │ write session/CONTROLLER_FROZEN_N
          ▼
┌───────────────────┐
│ run_test.py       │ ──── exactly one 20-scene pass on manifests/test.json
│ (eval_locked.cmd) │
└───────────────────┘
```

**Reported path (round 3):**

| Stage | Result |
|-------|--------|
| Explore 0–23 | 18 / 24 |
| Held-out explore 24–43 (pre-freeze check) | 13 / 20 |
| Locked `test.json` after freeze 3 | **15 / 20** |

Rounds 1–2 scored 4/20 and then improved on explore; only freeze 3 is the
published locked place rate. Open/close-only **16 / 20** (`open_close.py`) is a
**different** task and must not be mixed with the place figure.

---

## 7. Implementation details worth knowing

### Camera / depth contract

- Control cameras: `top` (RGB + depth), `side` (RGB), `wrist` (RGB).
- `front` / `overview` may exist in the MJCF for viewing; they are not control inputs.
- Depth is metric along the camera ray; agent observations store mm as uint16.

### Why Session wraps the backend

Early MCP packing stripped camera arrays. `Session` therefore returns env
observations after `_strip_for_agent`, so explore code always sees numpy cameras
even when PNG packing is used for remote agents.

### Action dual mode

`RaxDrawerEnv.step` forwards to `SO101MujocoRobot.send_action`, which detects
velocity keys vs `*.pos` keys. The place controller deliberately mixes both:
velocity for compliant handle contact, position IK for pick accuracy.

### Instinct / headless GL

On MI210/MI300-class hosts prefer `MUJOCO_GL=osmesa` (or `egl` where it works)
for offscreen RGB-D. GIF render:

```bash
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.rax_place.render_place_gif \
  --split test --scene 1 --out labs/rax_drawer/media/open_place_close_rgbd.gif
```

### Contamination

`place_close.py` predated the Session. The frozen controller must not import it.
Prior human knowledge remains a bias risk; the audit log and “never tune on test”
rule still apply.

---

## 8. What was intentionally not used

- **SmolVLA / pi0.5 / other VLAs:** no drawer open→place demos in this scene;
  camera and action layouts do not match Lab 01 pick datasets.
- **Privileged scripted pull** that servos to a measured drawer-joint Y.
- **Retuning after reading locked failures** (protocol violation).

---

## 9. Reproduce

```bash
# Locked 15/20 replay (freeze 3)
./labs/rax_drawer/rax_place/eval_locked.cmd

# One episode RGB-D GIF
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.rax_place.render_place_gif

# Explore self-test helper
.venv-rocm/bin/python -m labs.rax_drawer.rax_place.run_explore
```

Success definition and scene physics live in `environment.py`. Tool surface for
agents lives in `tools.py` / `backend.py`. Behavior of the frozen skill lives
only under `code/open_place_close/`.
