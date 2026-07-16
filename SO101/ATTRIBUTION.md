# SO-101 Simulation Assets — Attribution

MuJoCo models, meshes, URDF/MJCF files, and scene definitions in this directory
are used by SO-101 SimStudio for simulation and dataset recording.

---

## Robot model lineage

### The Robot Studio — SO-ARM100 / SO-101

The SO-101 arm design, bill of materials, and 3D-printable parts are documented
in:

- **Repository:** https://github.com/TheRobotStudio/SO-ARM100  
- **License:** Apache-2.0  

LeRobot's SO-101 hardware guide references this repository as the source for
parts and assembly. Visual and collision meshes in this tree follow that robot
family's geometry.

### Hugging Face LeRobot

SO-101 simulation conventions (joint naming, calibration XML, Feetech motor
classes, dataset integration) align with upstream LeRobot:

- **Repository:** https://github.com/huggingface/lerobot (submodule at `lerobot/`)  
- **License:** Apache-2.0  
- **Docs:** `lerobot/src/lerobot/robots/so_follower/so101.md`

This project's `so101_new_calib.xml` / `so101_new_calib.urdf` and related assets
are maintained for MuJoCo teleoperation in SimStudio and may include local
changes (collision decomposition, scene wiring) on top of LeRobot-compatible
structure.

### CAD export metadata

`so101_new_calib.xml` includes a header noting generation via **onshape-to-robot**
from an Onshape document. If you redistribute modified CAD-derived meshes,
ensure your use complies with the original CAD export and any third-party CAD
platform terms applicable to your workflow.

---

## Local modifications in this repository

| Modification | Location | Notes |
|--------------|----------|-------|
| Convex collision hulls (CoACD) | `assets/collision/*_hull_*.stl` | Gripper collision fix; see [asset_processing.md](asset_processing.md) |
| Scene definitions | `scenes/<scene_id>/` | Project task layouts (e.g. `simple_pick`) |
| Joint / motor defaults | `joints_properties.xml`, MJCF `<default>` blocks | Tuned for sim + LeRobot parity |

CoACD reference: https://colin97.github.io/CoACD/

---

## Scenes

Per-scene MJCF under `scenes/` (table, cube, cameras) is **project-authored**
unless otherwise noted in the scene directory. Default scene: `simple_pick`.

---

## License summary

| Content | License |
|---------|---------|
| Original SimStudio scene and project-specific MJCF edits | Apache-2.0 (see root [LICENSE](../LICENSE)) |
| SO-101 robot design lineage (SO-ARM100) | Apache-2.0 ([TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)) |
| LeRobot integration patterns and upstream SO-101 docs | Apache-2.0 ([huggingface/lerobot](https://github.com/huggingface/lerobot)) |

For full license text, see [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md).
For acknowledgements, see [ACKNOWLEDGEMENTS.md](../ACKNOWLEDGEMENTS.md).
