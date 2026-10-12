# MuJoCo scenes

Each subdirectory is one simulation scene (robot + workspace + task objects).

| Scene | Path | Description |
|-------|------|-------------|
| `simple_pick` | `simple_pick/scene.xml` | Hello-world: single cube pick on a table |
| `kitchen_counter` | `kitchen_counter/scene.xml` | Text2MuJoCo 01 workcell + SO-101: cube, button, place box |
| `smart_drawer` | `smart_drawer/scene.xml` | Text2MuJoCo 02 workcell + SO-101: sliding tool drawer |

Robot arm MJCF is included from `SO101/so101_new_calib.xml` (shared across scenes).

## Cameras (record vs teleop vs train)

| Role | Mechanism | Default |
|------|-----------|---------|
| **Recorded / dataset** | `robot.camera_names` in the YAML | `[front, top, wrist]` |
| **Teleop GLFW 4th pane** | MJCF cam named `overview` (auto-shown if present) | Observation only — **not** recorded unless listed in `camera_names` |
| **Train / eval subset** | LeRobot `--rename_map` (dataset key → policy key) | Map only the cams the policy should use |

**Record more views** (e.g. keep overview for later experiments):

```yaml
robot:
  camera_names: [front, top, wrist, overview]
```

Keys in the dataset become `observation.images.camera_<name>` (e.g.
`camera_wrist`, `camera_overview`).

**Train on a free subset** without deleting videos from the dataset — pass a
`rename_map` that includes only the cams you want. Unmapped image keys are not
fed as that policy’s visual inputs. Camera **count** must match the policy
(e.g. SmolVLA base expects three named slots; MolmoAct2 / some VLAs expect two).

Example — train on gripper + overview only (2-cam policy, `cam0` / `cam1`):

```bash
--rename_map='{"observation.images.camera_wrist":"observation.images.cam0","observation.images.camera_overview":"observation.images.cam1"}'
```

Example — Lab 01 SmolVLA (3 cams → `camera1/2/3`):

```bash
--rename_map='{"observation.images.camera_top":"observation.images.camera1","observation.images.camera_front":"observation.images.camera2","observation.images.camera_wrist":"observation.images.camera3"}'
```

Lab scripts set these via `_env.sh` (`LAB01_RENAME_MAP`, `LAB01_MOLMO_RENAME_MAP`,
…). Override the env var or pass `--rename_map=...` on the train command.

Unused recorded streams stay on disk (no automatic strip). Prefer recording only
what you need if storage or decode cost matters; use `overview` in
`camera_names` only when a training recipe will select it.

Each scene directory needs an `assets` symlink to the shared meshes:

```bash
ln -sf ../../assets simple_pick/assets
```

MuJoCo resolves `meshdir` relative to the scene XML path; the symlink keeps nested scenes working without duplicating STL files.

Add new scenes as siblings, e.g. `kitchen_counter/scene.xml`, with layout data under `configs/scenes/<scene_id>/`.
