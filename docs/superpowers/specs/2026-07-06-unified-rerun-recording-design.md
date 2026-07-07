# Unified Rerun / MuJoCo Recording Design

**Date:** 2026-07-06  
**Status:** Reviewed — latency analysis added 2026-07-06; ready for implementation plan  
**Scope:** SO-101 SimStudio recording with keyboard, Joy-Con, and leader-arm teleop

---

## 1. Summary

Add a `--view_mode` parameter to the SimStudio record wrapper so operators can choose between **MuJoCo window** or **LeRobot official Rerun** visualization at recording time. Both modes produce identical **LeRobot dataset v3.0** output suitable for downstream policy training.

**Principles:**

- Rerun integration uses LeRobot's built-in `display_data` / `display_mode=rerun` path — no custom `rr.log()` patches.
- Display mode is orthogonal to dataset writing; `dataset.add_frame()` always runs.
- Teleop-specific code is limited to the keyboard listener conflict fix.
- The LeRobot submodule remains unmodified.

---

## 2. Goals

| Goal | Success criteria |
|------|------------------|
| Unified display toggle | `--view_mode mujoco \| rerun` works for all three teleop configs |
| v3.0 dataset output | Recorded datasets have `codebase_version: v3.0`, pass `validate_dataset` |
| LeRobot-native Rerun | Uses `init_visualization` / `log_visualization_data` from submodule |
| Keyboard recording control | Arrow keys + ESC control episodes without conflicting with movement keys |
| No submodule edits | All changes in `src/simstudio/scripts/record.py` and docs/configs |

## 3. Non-Goals

- Custom post-action Rerun patching (1-frame display lag fix) — deferred; not default.
- `view_mode=both` (MuJoCo + Rerun simultaneously) — out of scope for v1; can add later.
- Rerun support in `teleoperate.py` (leader preview without recording) — separate follow-up.
- Foxglove visualization — LeRobot supports it, but not required for this spec.
- macOS / CUDA environment support.

---

## 4. Background

### 4.1 Current state

- `record.py` delegates to `lerobot_record.record()` with plugin registration only.
- A prior implementation added `view_mode`, custom Rerun patching, and evdev keyboard support; it was reverted (`76a63e5`) due to keyboard key conflicts, display jitter, and complexity.
- Config templates set `display_data: false`; Rerun never activates.
- Test scripts (`test_keyboard_record.sh`) pass `--view_mode`, but the flag is currently ignored.
- Keyboard recording control (arrow keys / ESC) is broken: `_recording_events` is never linked to LeRobot's `events` dict.

### 4.2 LeRobot record loop (relevant excerpt)

```
get_observation() → build observation_frame
get_action()      → send_action()
add_frame()       → always writes to dataset
log_visualization_data() → only when display_data=true
```

Visualization happens **after** dataset write and does not affect saved frames.

### 4.3 Teleop action semantics (unchanged)

| Teleop | Action format | Robot mode |
|--------|---------------|------------|
| Keyboard | velocity `{vx, vy, vz, wrist_flex_rate, yaw_rate, gripper_delta}` | `action_mode: velocity` |
| Joy-Con | velocity (same keys) | `action_mode: velocity` |
| Leader arm | position `{*.pos}` | `action_mode: position` |

Dataset schema is determined by robot observation/action features, not by display mode.

---

## 5. Design

### 5.1 `view_mode` parameter

New CLI flag on `simstudio.scripts.record`:

```
--view_mode {mujoco,rerun}   default: mujoco
```

Mapping to LeRobot / robot config flags (applied in wrapper before calling `lerobot_record`):

| `view_mode` | `--display_data` | `--display_mode` | `--robot.render_window` |
|-------------|------------------|------------------|-------------------------|
| `mujoco` | `false` | (ignored) | `true` |
| `rerun` | `true` | `rerun` | `false` |

**Precedence:** Explicit CLI flags (`--display_data`, `--robot.render_window`, etc.) passed by the user override `view_mode` defaults. The wrapper strips `--view_mode` from argv before forwarding to LeRobot.

**Optional YAML default** (documentation only; wrapper reads CLI first):

```yaml
# configs/so101_mujoco_keyboard.yaml (comment block)
# Recommended: pass --view_mode rerun or --view_mode mujoco on CLI
```

We do not add `view_mode` to LeRobot's `RecordConfig` dataclass; it stays a SimStudio wrapper concern.

### 5.2 Architecture

```
User CLI: --config ... --view_mode rerun|mujoco
                    │
                    ▼
┌──────────────────────────────────────────────────────────┐
│  simstudio.scripts.record (wrapper)                     │
│                                                          │
│  1. Register SO-101 plugins                              │
│  2. Parse --view_mode → inject display/render flags      │
│  3. [keyboard only] Patch init_keyboard_listener         │
│     + SO101KeyboardTeleop.connect for recording events   │
│  4. Forward argv to lerobot_record                       │
└──────────────────────────┬───────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────┐
│  lerobot.scripts.lerobot_record (submodule, unmodified)  │
│                                                          │
│  init_visualization()  ← only when display_data=true     │
│  record_loop():                                          │
│    teleop.get_action() → robot.send_action()             │
│    dataset.add_frame()   ← always                       │
│    log_visualization_data() ← when display_data=true     │
└──────────────────────────────────────────────────────────┘
```

### 5.3 Rerun visualization (official path)

When `view_mode=rerun`:

1. Wrapper sets `--display_data true --display_mode rerun --robot.render_window false`.
2. Wrapper patches `log_rerun_data` for streaming (no `static=True`; see §5.3.1).
3. LeRobot calls `init_visualization("rerun", session_name="recording")` which spawns the Rerun viewer.
4. Each record loop iteration calls the patched `log_visualization_data()` with processed observation and action.

#### 5.3.1 Rerun streaming patch (SimStudio wrapper)

LeRobot upstream `log_rerun_data()` logs camera images with `static=True`, which prevents live frame updates in the Rerun viewer during recording (camera panes stay black). The record wrapper replaces this function with a streaming-compatible implementation that:

- Calls `rr.set_time("frame", sequence=...)` each record step
- Logs `rr.Image(...)` **without** `static=True`
- Reuses LeRobot blueprint layout via `_ensure_blueprint`

This keeps `init_visualization` / `shutdown_visualization` on the official path while fixing live preview. Submodule remains unmodified.

**Dependencies:** `rerun-sdk` via `lerobot[viz]` extra (already in ROCm setup).

**Remote Rerun server:** Existing LeRobot flags remain available:

```bash
--display_ip <host> --display_port <port>
```

### 5.4 MuJoCo visualization

When `view_mode=mujoco`:

1. Wrapper sets `--display_data false --robot.render_window true`.
2. MuJoCo GLFW window renders the simulation scene (existing robot behavior).
3. No Rerun process is started.

### 5.5 Keyboard teleop patch (only teleop-specific code)

**Problem:** LeRobot's `init_keyboard_listener()` registers a TerminalKeyListener for `n`/`r`/`q`/ESC. SO101 keyboard teleop uses pynput for WASD movement keys and arrow keys for recording control. Two listeners conflict.

**Solution:** When `teleop.type == so101_keyboard`:

1. **Module-level shared events dict** (created at import time):

   ```python
   _recording_events = {
       "exit_early": False,
       "rerecord_episode": False,
       "stop_recording": False,
   }
   ```

2. **Patch `init_keyboard_listener`** to return `(None, _recording_events)` — no TerminalKeyListener.

3. **Patch `SO101KeyboardTeleop.connect`** to call `self.set_recording_events(_recording_events)` after the original connect. This works regardless of call order because the dict exists before `teleop.connect()` runs (LeRobot calls connect before init_keyboard_listener).

4. **Recording key mapping** (existing teleop behavior, unchanged):

   | Key | Event |
   |-----|-------|
   | Right arrow | `exit_early = True` (save episode) |
   | Left arrow | `rerecord_episode = True`, `exit_early = True` |
   | ESC | `stop_recording = True`, `exit_early = True` |

5. **Disabled shortcuts:** LeRobot's `n`/`r`/`q` terminal shortcuts are not available during keyboard recording (documented).

**Joy-Con and leader arm:** Use LeRobot's default `init_keyboard_listener()` for `n`/`r`/`q`/ESC recording control. No teleop-specific patches.

### 5.6 Dataset output (v3.0)

Both view modes produce the same dataset structure:

```
datasets/<name>/
├── meta/
│   ├── info.json          # codebase_version: "v3.0"
│   ├── episodes/
│   └── stats.json
├── data/
│   └── chunk-*/file-*.parquet
└── videos/
    └── observation.images.*/chunk-*/file-*.mp4
```

**Per-frame content:**

- `observation.images.front`, `.top`, `.wrist` — encoded video streams
- `observation.state` — joint positions
- `action` — teleop commands (velocity or position keys)
- `task` — task description string from config

**Validation:** `uv run python -m simstudio.scripts.validate_dataset --root <dataset_root>`

**Training:** Compatible with `lerobot_train` and pinned submodule commit `c746ca2d`.

### 5.7 Display timing: why Rerun feels laggier than MuJoCo

This section documents observed behavior from prior keyboard + rerun experiments. It applies to **all teleops**, not keyboard alone.

#### Record loop order (LeRobot, unmodified)

```
1. obs  = robot.get_observation()   # offscreen camera render × N
2. act  = teleop.get_action()
3. robot.send_action(act)           # physics steps; GLFW render if render_window=true
4. dataset.add_frame(...)           # always
5. log_visualization_data(obs)      # only when display_data=true; uses obs from step 1
```

#### MuJoCo window path (`view_mode=mujoco`)

- `_render_glfw()` runs at the **end** of `send_action()`, after physics steps.
- The operator sees the **post-action** state in the same loop iteration as the key press.
- Display path: OpenGL → `swap_buffers` (low latency).

#### Rerun path (`view_mode=rerun`)

- `log_visualization_data()` receives `obs_processed` captured in **step 1**, before `send_action()`.
- The operator sees the **pre-action** state — at least one record frame (~33 ms at 30 Hz) behind input.
- Display path: offscreen render → NumPy → Rerun IPC → viewer decode/UI (higher latency).
- `render_window=false` removes the low-latency GLFW fallback.

#### Additional overhead in rerun mode

Each record frame pays for:

1. **N offscreen camera renders** in `get_observation()` (default: front + top + wrist at 640×480).
2. **Rerun logging** of all observation images and scalars via `log_rerun_data()`.
3. **Loop slowdown** on CPU-only hosts — if the loop drops below target FPS, lag compounds.

#### Factors in the reverted implementation (76a63e5^) that made rerun worse

These are **explicitly excluded** from v1; listed here to explain past experience:

| Factor | Effect |
|--------|--------|
| Keyboard low-pass filter (`alpha=0.4`) | Smoothed velocity commands; motion felt sluggish |
| Custom `rr.log()` in `send_action` + official `log_visualization_data` | Double logging; timing confusion |
| Extra `get_observation()` inside send_action patch | Additional camera renders per frame |
| `static=True` on Rerun images | Suboptimal update semantics in viewer |

#### Impact on dataset vs preview

| Concern | Affected by display lag? |
|---------|--------------------------|
| Saved parquet / video frames | No — `add_frame` uses the same pipeline in both modes |
| Live operator preview in Rerun | Yes — expect noticeably higher latency than MuJoCo window |
| Policy training on recorded data | No |

**Recommendation for operators:** Use `view_mode=mujoco` when low-latency visual feedback matters during teleop. Use `view_mode=rerun` when camera-centric monitoring (e.g. wrist view) is preferred and ~1 frame lag is acceptable.

---

## 6. User-facing commands

### 6.1 Recording with MuJoCo window (default)

```bash
# Keyboard
uv run python -m simstudio.scripts.record \
  --config configs/so101_mujoco_keyboard.yaml \
  --view_mode mujoco

# Joy-Con (right)
uv run python -m simstudio.scripts.record \
  --config configs/so101_mujoco_joycon.yaml \
  --view_mode mujoco

# Leader arm
uv run python -m simstudio.scripts.record \
  --config configs/so101_mujoco_leader.yaml \
  --view_mode mujoco
```

### 6.2 Recording with Rerun

```bash
uv run python -m simstudio.scripts.record \
  --config configs/so101_mujoco_keyboard.yaml \
  --view_mode rerun
```

(Same pattern for joycon / leader configs.)

### 6.3 Recording controls by teleop

| Control | Keyboard | Joy-Con | Leader |
|---------|----------|---------|--------|
| Save episode | Right arrow | A (right) / Left d-pad (left) | `n` or Right arrow* |
| Rerecord | Left arrow | Y (right) / Up d-pad (left) | `r` |
| Stop all | ESC | Plus / Minus | `q` or ESC* |

\* Leader / Joy-Con also accept LeRobot terminal shortcuts when the record terminal has focus.

---

## 7. Implementation plan (files to change)

| File | Change |
|------|--------|
| `src/simstudio/scripts/record.py` | Add `_detect_view_mode()`, argv rewriting, keyboard patches |
| `configs/so101_mujoco_keyboard.yaml` | Document `--view_mode` in header comments |
| `configs/so101_mujoco_joycon.yaml` | Same |
| `configs/so101_mujoco_leader.yaml` | Same |
| `test_keyboard_record.sh` | Default `VIEW_MODE=rerun` or keep `mujoco`; verify flag works |
| `test_joycon_record.sh` | Add optional `VIEW_MODE` arg |
| `AGENTS.md` | Update `view_mode` docs: two modes, default `mujoco` |
| `QUICKSTART.md` | Add rerun recording examples |

**No changes to:**

- `lerobot/` submodule
- `so101_keyboard/teleop_so101_keyboard.py` (connect patch is in record.py)
- `so101_joycon/`, `so101_leader/` teleoperator implementations
- Robot MuJoCo implementation

---

## 8. Testing

### 8.1 Smoke tests (manual)

| # | Test | Pass criteria |
|---|------|---------------|
| 1 | Keyboard + `--view_mode mujoco` | MuJoCo window visible; dataset saved; arrow keys control episodes |
| 2 | Keyboard + `--view_mode rerun` | Rerun viewer shows cameras; no MuJoCo window; dataset saved |
| 3 | Joy-Con + `--view_mode rerun` | Rerun viewer works; n/r/q or Joy-Con buttons control episodes |
| 4 | Leader + `--view_mode rerun` | Rerun viewer works; leader motion recorded as position actions |
| 5 | `validate_dataset` on each output | No errors; fps and action ranges OK |
| 6 | Headless CI | `--view_mode mujoco --robot.render_window false` (existing test config) still passes |

### 8.2 Automated tests

Add unit tests for `record.py` helper functions (no MuJoCo/Rerun runtime required):

- `_detect_view_mode()` parses CLI correctly; defaults to `mujoco`.
- `_apply_view_mode(argv, "rerun")` injects correct `--display_data`, `--display_mode`, `--robot.render_window` flags.
- `_apply_view_mode(argv, "mujoco")` sets `display_data=false`, `render_window=true`.
- User-supplied `--display_data` is not overwritten when explicitly present.

### 8.3 Regression guard

Do **not** reintroduce:

- Custom `rr.log()` in `send_action` monkey-patch
- evdev keyboard listener (unless separately requested for Wayland)
- Low-pass filter on keyboard actions

---

## 9. Known limitations

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| Rerun pre-action observation (§5.7) | Live preview lags input by ≥1 record frame; feels worse than MuJoCo | Accept for v1; use `mujoco` for low-latency teleop; see §10 optional optimizations |
| Rerun IPC + multi-camera render cost | Loop may run below 30 Hz on CPU-only; lag compounds | GPU recommended; reduce camera count/resolution; fall back to `mujoco` |
| Keyboard: no n/r/q shortcuts | Must use arrow keys | Document in QUICKSTART |
| pynput on pure Wayland | Keyboard teleop may not capture keys | Existing limitation; evdev is a separate future task |
| Submodule upgrade | API changes to `lerobot_record` could break wrapper | Pin submodule; test on upgrade branch |

---

## 10. Future extensions (out of scope for v1)

### 10.1 General

- `view_mode=both` — MuJoCo window + Rerun simultaneously
- Rerun in `teleoperate.py` (non-recording preview)
- evdev keyboard backend for Wayland-native input
- Default `view_mode: rerun` in config templates once validated on ROCm hardware

### 10.2 Optional Rerun latency optimizations (opt-in, post-v1)

If official Rerun preview latency remains unacceptable after v1:

| Optimization | Description | Trade-off |
|--------------|-------------|-----------|
| `--rerun_sync_actions` | Log cameras from a post-`send_action` observation fetch; single logging path only | Wrapper complexity; extra render pass; must not double-log with LeRobot |
| Front camera only | Log `observation.camera_front` to Rerun instead of all cameras | Loses top/wrist in live preview (still in dataset) |
| Lower preview resolution | Render/log at 320×240 for Rerun while dataset stays 640×480 | Requires separate preview render path |
| Frame-index time base | Use episode frame index for `rr.set_time` instead of wall clock | Better sync in viewer timeline |

Do **not** reintroduce keyboard low-pass filtering as a latency fix — it adds input smoothing lag unrelated to display path.

---

## 11. Decision log

| Decision | Rationale |
|----------|-----------|
| Use LeRobot official Rerun, not custom `rr.log()` | Stays compatible with submodule upgrades and Isaac/LeRobot ecosystem |
| `view_mode` in wrapper only | Avoids forking LeRobot RecordConfig |
| Default `mujoco` | Safer for CI/headless; explicit opt-in to Rerun |
| Shared `_recording_events` dict for keyboard | Works regardless of connect/listener init order |
| No `both` mode in v1 | Reduces GPU load confusion; user chose binary selection |
| Do not modify submodule | Project convention per AGENTS.md |
| No keyboard low-pass filter | Prior `alpha=0.4` filter added input lag; unrelated to display fix |
| Default `mujoco` over `rerun` | MuJoCo window is post-action and lower latency for teleop (§5.7) |

---

## 12. Acceptance criteria

Implementation is complete when:

1. `--view_mode rerun` and `--view_mode mujoco` work for keyboard, Joy-Con, and leader recording configs.
2. Datasets from both modes pass `validate_dataset` with v3.0 metadata.
3. Keyboard arrow-key / ESC recording control works in both modes.
4. Joy-Con and leader retain LeRobot n/r/q recording shortcuts.
5. Unit tests for view_mode argv rewriting pass in `make test`.
6. AGENTS.md and QUICKSTART.md reflect the two-mode behavior.
