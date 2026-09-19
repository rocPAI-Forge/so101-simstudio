---
license: other
license_name: nvidia-open-model-license
license_link: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
library_name: lerobot
pipeline_tag: robotics
tags:
  - robotics
  - lerobot
  - groot
  - gr00t
  - so101
  - simstudio
datasets:
  - alexhegit/so101-simstudio-lab01-pnp
base_model: nvidia/GR00T-N1.7-3B
---

# SO-101 Lab01 Pick-and-Place — GR00T N1.7 (negative result)

Fine-tune of [`nvidia/GR00T-N1.7-3B`](https://huggingface.co/nvidia/GR00T-N1.7-3B) on the Lab 01 **50-episode** joint-position dataset. **This is a documented negative result**, not a classroom pick-and-place policy. Same data: ACT 6-D **~58%** and MolmoAct2 6-D **~54%** full-range; this checkpoint **does not lift the cube**.

**Training data:** [alexhegit/so101-simstudio-lab01-pnp](https://huggingface.co/datasets/alexhegit/so101-simstudio-lab01-pnp)

This Hub revision is the **kept Lab 01 GR00T run**: `embodiment_tag=new_embodiment`, 6-D `*.pos` (pad to GR00T `max_action_dim`), relative actions with gripper excluded, LLM and vision **frozen**, batch **4** × **10 000** steps on AMD Instinct MI300X (~35 min, train loss ~0.03–0.1). Checkpoint `010000`.

**Do not use** the later continuation 30K (`10K` as `base_model_path` after cosine lr=0) or visual-unfreeze 20K — both scored **0/10** with **worse** approach than this 10K (`reach_8cm` 9–10/10 → 0–2/10). Details: SimStudio [`groot_n1_7_adapt.md`](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/groot_n1_7_adapt.md), Lab 01 §6.4, and [issue #6](https://github.com/rocPAI-Forge/so101-simstudio/issues/6).

Cameras stay Lab 01 keys (`camera_front`, `camera_top`, `camera_wrist`). Relative actions **require RTC** (`GrootPolicy.select_action` is not implemented). Eval in this repo patches gated Cosmos tokenizer files to public `Qwen/Qwen3-VL-2B-Instruct`.

Weights inherit the **NVIDIA Open Model License** from the N1.7 base (not Apache-2.0).

## Closed-loop (do not treat as a working policy)

Task: pick the cube and place it in the box. `reset_arm: home`, RTC `execution_horizon=8`. Cube stays near spawn `z≈0.015` (no lift).

| Protocol | Backend | Success | Funnel (reach 8 cm → 4 cm → close → lift) | Real chunk Hz |
|----------|---------|---------|-------------------------------------------|---------------|
| Full-range random | 8060S EGL | **0/10** | 9/10 → 7/10 → 5/10 → **0/10** | **2.53** (~395 ms) |
| Fixed demo cube | 8060S EGL | **0/10** | 9/10 → 6/10 → 2/10 → 0/10 | **2.51** |
| Full-range random | MI300X OSMesa | **0/10** | 10/10 → 6/10 → 1/10 → 0/10 | control compute ~2 Hz (software render) |

Log “10k+ Hz” figures are RTC queue pops, not `predict_action_chunk`.

## Documentation

| Resource | Link |
|----------|------|
| SimStudio repo | [rocPAI-Forge/so101-simstudio](https://github.com/rocPAI-Forge/so101-simstudio) |
| Lab 01 GR00T notes | [labs/lab01_pnp/groot_n1_7_adapt.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/groot_n1_7_adapt.md) |
| Lab 01 §6.4 | [labs/lab01_pnp/lab01_pnp.md](https://github.com/rocPAI-Forge/so101-simstudio/blob/main/labs/lab01_pnp/lab01_pnp.md) |
| Base weights | [nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B) |

## Eval (SimStudio)

```bash
source .venv-rocm/bin/activate
LAB01_POLICY_PATH=./outputs/hub/lab01_pnp_groot_n17 \
LAB01_EVAL_CONFIG=labs/lab01_pnp/configs/rollout_groot.yaml \
LAB01_EVAL_EPISODES=10 LAB01_MUJOCO_GL=egl LAB01_RENDER_WINDOW=false \
  ./labs/lab01_pnp/eval.cmd
```
