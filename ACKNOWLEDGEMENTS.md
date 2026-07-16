# Acknowledgements

SO-101 SimStudio builds on the work of many open-source projects and communities.
We are grateful to the authors and maintainers listed below.

Legal license terms for each component are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)
and [SO101/ATTRIBUTION.md](SO101/ATTRIBUTION.md).

---

## Core frameworks

| Project | Authors / org | Role in SimStudio |
|---------|---------------|-------------------|
| [**LeRobot**](https://github.com/huggingface/lerobot) | Hugging Face | Dataset format (v3.0), recording/replay pipeline, robot & teleoperator plugin APIs |
| [**MuJoCo**](https://github.com/google-deepmind/mujoco) | Google DeepMind | Physics simulation, offscreen rendering, GLFW viewer |
| [**PyTorch**](https://github.com/pytorch/pytorch) | PyTorch team | Tensor backend (ROCm build in this project) |

## Robot design & simulation assets

| Project | Authors / org | Role in SimStudio |
|---------|---------------|-------------------|
| [**SO-ARM100 / SO-101**](https://github.com/TheRobotStudio/SO-ARM100) | The Robot Studio | Open-hardware arm design; mesh and URDF lineage for the simulated SO-101 |
| [**CoACD**](https://github.com/SarahWeiii/CoACD) | CoACD authors | Approximate convex decomposition for gripper collision geometry |

## Teleoperation & input

| Project | Authors / org | Role in SimStudio |
|---------|---------------|-------------------|
| [**joycon-robotics**](https://github.com/box2ai-robotics/joycon-robotics) | box2ai-robotics | Joy-Con HID integration (optional; patched at install time) |
| [**python-evdev**](https://github.com/gvalkov/python-evdev) | gvalkov et al. | Focus-independent Linux keyboard recording controls |
| [**pynput**](https://github.com/moses-palmer/pynput) | Moses Palmer et al. | Keyboard teleop fallback when evdev is unavailable |

## Visualization

| Project | Authors / org | Role in SimStudio |
|---------|---------------|-------------------|
| [**Rerun**](https://github.com/rerun-io/rerun) | Rerun | Live multi-camera viewer during recording (`--view_mode rerun`) and dataset visualization |

## Python ecosystem (selected)

| Project | Role |
|---------|------|
| [**NumPy**](https://github.com/numpy/numpy) / [**SciPy**](https://github.com/scipy/scipy) | Numerics, Jacobian IK |
| [**Hugging Face Datasets**](https://github.com/huggingface/datasets) | Dataset I/O via LeRobot |
| [**matplotlib**](https://github.com/matplotlib/matplotlib) | Plotting (via LeRobot viz extras) |

## References & tools cited in docs

| Resource | Notes |
|----------|-------|
| [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) | CAD → URDF export pipeline (see `SO101/so101_new_calib.xml` header) |
| [obj2mjcf](https://github.com/kevinzakka/obj2mjcf) | Referenced in collision-geometry notes |
| [MuJoCo mesh collision discussion](https://github.com/google-deepmind/mujoco/issues/436) | Background for gripper convex decomposition |

## Hardware communities

| Community | Notes |
|-----------|-------|
| **Feetech STS3215** servo ecosystem | Leader-arm teleop uses LeRobot's SO leader stack and Feetech motor conventions |
| **Nintendo Switch Joy-Con** | Consumer hardware; not affiliated with Nintendo. Joy-Con support via third-party HID libraries |

---

If we missed a project you maintain or depend on, please open an issue or pull request.
