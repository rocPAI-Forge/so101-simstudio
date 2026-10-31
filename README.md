<p align="center">
  <img src="docs/assets/banner-flat-logo.png" alt="SO-101 SimStudio — MuJoCo · LeRobot · Expert Trajectories" width="900"/>
</p>

# SO-101 SimStudio

MuJoCo + LeRobot studio for SO-101 expert trajectories, closed-loop policy eval, and RAX-style tool-only control.

<p align="center">
  <img src="docs/assets/architecture-flow.png" alt="Teleoperation → MuJoCo → LeRobot dataset v3.0" width="820"/>
</p>

**Platform:** Ubuntu 24.04 + AMD ROCm 7.x only (`make rocm-sync`). CUDA and macOS are not supported yet — [ROADMAP](ROADMAP.md).

## What you can do

| Capability | Practice | Docs |
|------------|----------|------|
| Teleop record (keyboard / Joy-Con / leader) | MuJoCo or Rerun GUI → LeRobot v3.0 | [QUICKSTART](QUICKSTART.md) |
| Dataset replay / validate / visualize | `replay`, `validate_dataset`, `dataset_viz` | [QUICKSTART](QUICKSTART.md) |
| Demo scenes | `simple_pick`, `kitchen_counter`, `smart_drawer` | [Scenes](SO101/scenes/README.md) |
| Lab 01 pick-and-place | Record or agent-collect → train → Home20 eval | [lab01_pnp](labs/lab01_pnp/lab01_pnp.md) · [Home20](labs/lab01_pnp_agent/home20.md) |
| RAX drawer (tool-only agent) | Explore → compile → locked test on `smart_drawer` | [rax_drawer](labs/rax_drawer/README.md) · [method](labs/rax_drawer/rax_place/RAX_METHOD.md) |
| External VLA inference | In-process adapter or sidecar | [external_vla](docs/external_vla.md) |

## Try it now

```bash
git clone --recursive https://github.com/rocPAI-Forge/so101-simstudio.git
cd so101-simstudio
make rocm-sync
source .venv-rocm/bin/activate
```

| Path | Command |
|------|---------|
| Keyboard record | `.venv-rocm/bin/python -m simstudio.scripts.record --config configs/so101_mujoco_keyboard.yaml` |
| Smoke teleop / record | `make smoke-keyboard-teleop` · `make smoke-keyboard-record VIEW_MODE=mujoco EPISODES=1` |
| Smart drawer teleop | `make smoke-keyboard-teleop-smart-drawer` |
| RAX place locked replay | `./labs/rax_drawer/rax_place/eval_locked.cmd` → **15 / 20** |
| Lab 01 Home20 | see [lab01_pnp](labs/lab01_pnp/lab01_pnp.md) / [home20.md](labs/lab01_pnp_agent/home20.md) |

Full install, Joy-Con, leader arm, and train/eval recipes: [QUICKSTART.md](QUICKSTART.md).

## Results at a glance

| Track | Headline | Details |
|-------|----------|---------|
| Lab 01 Home20 | ACT **53/60**, SmolVLA **51/60**, … (shared frozen poses; not a general ranking) | [home20.md](labs/lab01_pnp_agent/home20.md) |
| RAX open → close | **16 / 20** locked | [rax_drawer](labs/rax_drawer/README.md) |
| RAX open → place → close | **15 / 20** freeze 3 | [RAX_METHOD](labs/rax_drawer/rax_place/RAX_METHOD.md) · replay `eval_locked.cmd` |

## Documentation

- [Quick start](QUICKSTART.md) · [Agent / layout guide](AGENTS.md) · [Architecture](DESIGN.md) · [Roadmap](ROADMAP.md)
- Labs: [lab01_pnp](labs/lab01_pnp/lab01_pnp.md) · [lab01 agent + Home20](labs/lab01_pnp_agent/lab01_pnp_agent.md) · [rax_drawer](labs/rax_drawer/README.md)
- [External VLA](docs/external_vla.md) · [Acknowledgements](ACKNOWLEDGEMENTS.md) · [Third-party notices](THIRD_PARTY_NOTICES.md) · [Asset attribution](SO101/ATTRIBUTION.md)

## License

Project-authored code: [Apache-2.0](LICENSE). Bundled third-party code and assets keep their own licenses — see [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
