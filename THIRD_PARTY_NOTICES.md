# Third-Party Notices

This document lists third-party software and assets used by **SO-101 SimStudio**
(`so101-simstudio`). It supplements the project [LICENSE](LICENSE) (Apache-2.0,
Copyright 2026 Alex He).

For a readable thank-you list, see [ACKNOWLEDGEMENTS.md](ACKNOWLEDGEMENTS.md).

**Important:** The root `LICENSE` applies only to original work in this
repository. Third-party components remain under their own licenses. When you
redistribute or publish releases of this project, retain this file, the root
`NOTICE`, submodule licenses, and any copyright notices required by permissive
or copyleft dependencies you actually ship.

---

## 1. Git submodules (vendored repositories)

### 1.1 Hugging Face LeRobot

| Field | Value |
|-------|-------|
| **Repository** | https://github.com/huggingface/lerobot |
| **Path** | `lerobot/` |
| **Pinned commit** | `30da8e687a6dfc617fcd94afc367ac7071c376ce` (tag `v0.6.0`) |
| **License** | Apache-2.0 |
| **Copyright** | Copyright 2024 The Hugging Face team |
| **License file** | `lerobot/LICENSE` |
| **Usage** | Editable Python dependency (`lerobot[kinematics,dataset,viz,deepdiff-dep]`); dataset format, recording/replay pipeline, robot/teleoperator plugin APIs |

**Apache-2.0 compatibility:** Fully compatible with this project's Apache-2.0
license. Keep the submodule unmodified when possible; retain Hugging Face
copyright and license text in distributions that include `lerobot/`.

LeRobot pulls a large transitive dependency tree (PyTorch, Hugging Face
`datasets`, etc.). Those packages keep their own licenses; see
[Section 2](#2-direct-and-transitive-python-dependencies).

---

### 1.2 box2ai-robotics joycon-robotics

| Field | Value |
|-------|-------|
| **Repository** | https://github.com/box2ai-robotics/joycon-robotics |
| **Path** | `third_party/joycon-robotics/` |
| **Pinned commit** | `6f87e3b7055c1defd63d139df8b24c571825b938` |
| **License (Python package)** | MIT |
| **Copyright** | Copyright (c) 2026 box2ai-robotics |
| **License file** | `third_party/joycon-robotics/LICENSE` |
| **Usage** | Optional Joy-Con teleop; installed via `make joycon-sync` (`pip install -e` + local patch) |

**Project patch:** `patches/joycon-robotics.patch` applies SO-101-specific fixes
(serial compatibility, English connect messages) at install time. The patch is
project-authored; the underlying joycon-robotics code remains MIT-licensed.

**MIT compatibility:** MIT is compatible with Apache-2.0. You must include the
MIT copyright notice and permission text when distributing joycon-robotics
source or binaries.

#### Optional bundled components (not installed by default)

The joycon-robotics repository also contains **optional system-level** tools
for Linux/Windows Joy-Con pairing. **SO-101 SimStudio does not run these
installers** as part of `make joycon-sync` or normal recording. They are listed
here because the submodule source tree includes them.

| Component | Path (inside submodule) | License | Notes |
|-----------|-------------------------|---------|-------|
| `joycond` | `joyconrobotics/system_lib/joycond/` | GPL-3.0 | Userspace daemon for Joy-Con pairing on Linux |
| `dkms-hid-nintendo` | `joyconrobotics/system_lib/dkms-hid-nintendo/` | GPL-2.0 | Linux kernel driver (DKMS) |
| BetterJoy | `hidapi_for_windows/BetterJoy_v7.1/` | MIT | Windows HID helper (David Khachaturov, 2018) |

If you only use `make joycon-sync` (Python editable install + patch), the
**GPL components are not linked into or distributed as part of** the SimStudio
Python package. If you separately build or redistribute GPL kernel modules or
daemons from that submodule, comply with GPL-2.0 / GPL-3.0 independently.

---

## 2. Direct and transitive Python dependencies

Declared in [pyproject.toml](pyproject.toml) and resolved by `uv` / pip (including
LeRobot's dependency tree).

### 2.1 Direct dependencies (this project)

| Package | Role | Typical license | Notes |
|---------|------|-----------------|-------|
| **mujoco** | MuJoCo physics engine | Apache-2.0 | https://github.com/google-deepmind/mujoco |
| **lerobot** | Dataset + robot framework | Apache-2.0 | Local editable submodule; see §1.1 |
| **scipy** | Numerics (IK, etc.) | BSD-3-Clause | |
| **matplotlib** | Plotting (via LeRobot viz) | PSF-based / BSD-style | See package metadata for full text |
| **pynput** | Keyboard input fallback | **LGPL-3.0** | Used when evdev backend unavailable; dynamic import as separate library |
| **evdev** | Focus-independent recording keys (Linux) | BSD-3-Clause | Preferred input backend on Linux |

### 2.2 Major transitive dependencies (via LeRobot)

Non-exhaustive list of commonly installed packages:

| Package | Typical license | Notes |
|---------|-----------------|-------|
| **torch** | BSD-3-Clause | Deep learning backend for LeRobot policies / training |
| **numpy** | BSD-3-Clause | |
| **datasets** (Hugging Face) | Apache-2.0 | Dataset I/O |
| **pyarrow** | Apache-2.0 | Columnar data |
| **rerun-sdk** | MIT **OR** Apache-2.0 | Optional `--view_mode rerun` visualization |
| **huggingface-hub**, **safetensors**, etc. | Mostly Apache-2.0 / MIT | See installed package metadata |

To inspect licenses in your environment:

```bash
uv pip show mujoco lerobot torch pynput evdev
# or, for a full tree:
uv pip licenses  # if pip-licenses is installed
```

---

## 3. Simulation assets (`SO101/`)

Robot meshes, URDF/MJCF, collision hulls, and scene definitions live under
[`SO101/`](SO101/). See [SO101/ATTRIBUTION.md](SO101/ATTRIBUTION.md) for
provenance and license lineage.

Summary:

| Source | License | Relationship |
|--------|---------|--------------|
| [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) | Apache-2.0 | SO-101 hardware design and mesh lineage |
| [Hugging Face LeRobot](https://github.com/huggingface/lerobot) SO-101 docs / sim conventions | Apache-2.0 | Calibration, joint properties, integration patterns |
| Onshape → robot export (see XML header in `so101_new_calib.xml`) | — | CAD export metadata; verify Onshape export terms for your use case |
| CoACD convex decomposition | — | Algorithm used for gripper collision hulls; see `SO101/asset_processing.md` |
| Scene layouts (`SO101/scenes/`) | Apache-2.0 (this project) | Project-authored task scenes |

---

## 4. License compatibility with Apache-2.0 (summary)

| Category | Verdict | Action |
|----------|---------|--------|
| Apache-2.0 (LeRobot, MuJoCo, datasets, pyarrow, SO-ARM100 lineage) | Compatible | Retain copyright and license notices |
| MIT (joycon-robotics Python, BetterJoy) | Compatible | Include MIT notice in distributions |
| BSD-3-Clause / PSF (NumPy, SciPy, PyTorch, evdev) | Compatible | Retain copyright notices |
| LGPL-3.0 (pynput) | Generally acceptable as **separate** Python library dependency | Document in notices; provide LGPL text and source access per LGPL if you distribute binaries |
| GPL-2.0 / GPL-3.0 (optional joycon kernel tools) | **Not part of default install** | Do not combine into a single proprietary binary without separate GPL compliance; document if you ship them |

This table is practical guidance for open-source distribution, not legal advice.
Consult counsel for commercial or compliance-critical releases.

---

## 5. What to include in a release tarball or binary distribution

Minimum recommended bundle:

1. Root [LICENSE](LICENSE) and [NOTICE](NOTICE)
2. This file (`THIRD_PARTY_NOTICES.md`)
3. [SO101/ATTRIBUTION.md](SO101/ATTRIBUTION.md)
4. Submodule license files: `lerobot/LICENSE`, `third_party/joycon-robotics/LICENSE`
5. If Joy-Con support is included: MIT text from joycon-robotics
6. If keyboard teleop via pynput is included: LGPL-3.0 notice and pointer to pynput source

Git submodules are not always included in PyPI-style wheels; if you publish
wheels without submodules, document that users must clone with `--recursive` or
install LeRobot and joycon-robotics separately under their respective licenses.

---

## 6. Trademarks

Third-party names (MuJoCo, Hugging Face, LeRobot, Nintendo Switch, Joy-Con,
PyTorch, etc.) are trademarks of their respective owners. This project is not
affiliated with or endorsed by those entities.
