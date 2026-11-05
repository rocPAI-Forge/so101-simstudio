# RAX drawer → ACT / SmolVLA pipeline

Split of work:

| Host | Role |
|------|------|
| **halo** (`.13`, Radeon 8060S) | EGL collect + closed-loop eval |
| **DORobot** (MI300X) | Heavy ACT / SmolVLA training (no MuJoCo GL) |

**Depth ablation (place100d):** see [`DEPTH_ABLATION.md`](DEPTH_ABLATION.md).

## Hub assets (place100d ACT RGB / RGB-D)

| Asset | Path |
|-------|------|
| Dataset | [`alexhegit/so101-simstudio-rax-drawer-place100d`](https://huggingface.co/datasets/alexhegit/so101-simstudio-rax-drawer-place100d) |
| ACT RGB | [`alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgb-place100d) |
| ACT RGB-D | [`alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d`](https://huggingface.co/alexhegit/so101-simstudio-rax-drawer-act-rgbd-place100d) |

Closed-loop test20: RGB **7/20**, RGB-D **11/20**.

## 1. Collect (~100 successes) on halo

```bash
cd ~/Repo/so101-simstudio
source .venv-rocm/bin/activate
MUJOCO_GL=egl RAX_TARGET=100 ./labs/rax_drawer/collect_rax_place.cmd
```

Dataset root: `datasets/so101-simstudio-rax-drawer-place100d/` (includes `camera_top_depth`).

## 2. Sync to DORobot (optional)

```bash
rsync -avP --delete \
  ./datasets/so101-simstudio-rax-drawer-place100d/ \
  DORobot:/root/so101-simstudio-rax/datasets/so101-simstudio-rax-drawer-place100d/
```

## 3. Train ACT (RGB and/or RGB-D)

```bash
./labs/rax_drawer/vla/train_act_rgb.cmd     # drops depth
./labs/rax_drawer/vla/train_act_rgbd.cmd    # keeps depth
# legacy single-run helper still: ./labs/rax_drawer/vla/train_act.cmd
```

## 4. Eval on halo

```bash
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.eval_policy \
  --policy ./outputs/train/rax_drawer_act_rgbd/checkpoints/030000/pretrained_model \
  --episodes 20 --split test
```

## 5. Push Hub

```bash
# if huggingface.co is unreachable:
export HF_ENDPOINT=https://hf-mirror.com
./labs/rax_drawer/vla/push_place100d_dataset.cmd
./labs/rax_drawer/vla/push_act_models.cmd
```

## 6. SmolVLA (optional; prior line)

`./labs/rax_drawer/vla/train_smolvla.cmd` then eval with `--rename-map` from `_env.sh`.
