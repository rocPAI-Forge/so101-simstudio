# Knobs for RAX-drawer → ACT / SmolVLA (collect on halo, train on DORobot).
RAX_DATASET_NAME="${RAX_DATASET_NAME:-so101-simstudio-rax-drawer-place100b}"
RAX_DATASET_HF_USER="${RAX_DATASET_HF_USER:-alexhegit}"
RAX_DATASET_REPO_ID="${RAX_DATASET_REPO_ID:-${RAX_DATASET_HF_USER}/${RAX_DATASET_NAME}}"
RAX_DATASET_ROOT="${RAX_DATASET_ROOT:-./datasets/${RAX_DATASET_NAME}}"

RAX_ACT_OUTPUT="${RAX_ACT_OUTPUT:-./outputs/train/rax_drawer_act_state6}"
RAX_ACT_LOG="${RAX_ACT_LOG:-./outputs/train/rax_drawer_act_state6.log}"
RAX_ACT_JOB_NAME="${RAX_ACT_JOB_NAME:-rax_drawer_act_state6}"
RAX_ACT_BATCH_SIZE="${RAX_ACT_BATCH_SIZE:-32}"
RAX_ACT_STEPS="${RAX_ACT_STEPS:-30000}"
RAX_ACT_SAVE_FREQ="${RAX_ACT_SAVE_FREQ:-10000}"
RAX_ACT_STATE_DIM="${RAX_ACT_STATE_DIM:-6}"

RAX_SMOL_OUTPUT="${RAX_SMOL_OUTPUT:-./outputs/train/rax_drawer_smolvla}"
RAX_SMOL_LOG="${RAX_SMOL_LOG:-./outputs/train/rax_drawer_smolvla.log}"
RAX_SMOL_JOB_NAME="${RAX_SMOL_JOB_NAME:-rax_drawer_smolvla}"
RAX_SMOL_BATCH_SIZE="${RAX_SMOL_BATCH_SIZE:-16}"
RAX_SMOL_STEPS="${RAX_SMOL_STEPS:-20000}"
RAX_SMOL_SAVE_FREQ="${RAX_SMOL_SAVE_FREQ:-5000}"
# Dataset cameras: top / side / wrist → SmolVLA camera1/2/3
RAX_SMOL_RENAME_MAP="${RAX_SMOL_RENAME_MAP:-{\"observation.images.camera_top\":\"observation.images.camera1\",\"observation.images.camera_side\":\"observation.images.camera2\",\"observation.images.camera_wrist\":\"observation.images.camera3\"}}"

RAX_EVAL_EPISODES="${RAX_EVAL_EPISODES:-20}"
RAX_EVAL_SPLIT="${RAX_EVAL_SPLIT:-test}"
