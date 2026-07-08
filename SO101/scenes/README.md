# MuJoCo scenes

Each subdirectory is one simulation scene (robot + workspace + task objects).

| Scene | Path | Description |
|-------|------|-------------|
| `simple_pick` | `simple_pick/scene.xml` | Hello-world: single cube pick on a table |

Robot arm MJCF is included from `SO101/so101_new_calib.xml` (shared across scenes).

Each scene directory needs an `assets` symlink to the shared meshes:

```bash
ln -sf ../../assets simple_pick/assets
```

MuJoCo resolves `meshdir` relative to the scene XML path; the symlink keeps nested scenes working without duplicating STL files.

Add new scenes as siblings, e.g. `kitchen_counter/scene.xml`, with layout data under `configs/scenes/<scene_id>/`.
