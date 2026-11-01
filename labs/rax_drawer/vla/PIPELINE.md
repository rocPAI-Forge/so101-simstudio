# RAX drawer → ACT / SmolVLA pipeline

Split of work:

| Host | Role |
|------|------|
| **halo** (`192.168.1.43`, Radeon 8060S) | EGL collect + closed-loop eval |
| **DORobot** (MI300X) | ACT / SmolVLA training only (no MuJoCo GL) |

## 1. Collect (~100 successes) on halo

```bash
cd ~/Repo/so101-simstudio
source .venv-rocm/bin/activate
MUJOCO_GL=egl RAX_TARGET=100 ./labs/rax_drawer/collect_rax_place.cmd
```

Dataset: `datasets/so101-simstudio-rax-drawer-place100/`

## 2. Sync to DORobot

```bash
rsync -avP --delete \
  ./datasets/so101-simstudio-rax-drawer-place100/ \
  DORobot:/root/so101-simstudio-rax/datasets/so101-simstudio-rax-drawer-place100/
```

## 3. Train ACT 6-D on MI300X

```bash
ssh DORobot 'cd /root/so101-simstudio-rax && source .venv-rocm/bin/activate && \
  RAX_ACT_BATCH_SIZE=64 RAX_ACT_STEPS=30000 \
  ./labs/rax_drawer/vla/train_act.cmd'
```

## 4. Eval ACT on halo

```bash
rsync -avP DORobot:/root/so101-simstudio-rax/outputs/train/rax_drawer_act_state6/ \
  ./outputs/train/rax_drawer_act_state6/
MUJOCO_GL=egl .venv-rocm/bin/python -m labs.rax_drawer.vla.eval_policy \
  --policy ./outputs/train/rax_drawer_act_state6/checkpoints/030000/pretrained_model \
  --episodes 20 --split test
```

## 5. SmolVLA (after ACT baseline)

Train on DORobot with `./labs/rax_drawer/vla/train_smolvla.cmd`, then eval on halo with `--rename-map` from `_env.sh`.
