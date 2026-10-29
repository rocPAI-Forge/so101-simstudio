# Agent-yaw / 193 / Home20 knobs.
# Sources the Lab 01 template, then sets only this line's pose file and Hub ids.
# Teleop dataset, train, and eval defaults stay in ../lab01_pnp/_env.sh.
_LAB01_AGENT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_LAB01_AGENT_DIR/../lab01_pnp/_env.sh"

# Lab01-PnP-Home20 pose manifest. Eval protocol, not a training dataset.
LAB01_HOME20_SPAWNS="${LAB01_HOME20_SPAWNS:-$_LAB01_AGENT_DIR/configs/smolvla_v4_div50_home20_spawns.json}"

# 193-episode follow2 set. On-disk observation.state stays 15-D.
LAB01_FOLLOW2_DATASET_NAME="${LAB01_FOLLOW2_DATASET_NAME:-so101-simstudio-lab01-pnp-agent-yaw-v4-actonly-follow2}"
LAB01_FOLLOW2_DATASET_ROOT="${LAB01_FOLLOW2_DATASET_ROOT:-./datasets/${LAB01_FOLLOW2_DATASET_NAME}}"
LAB01_FOLLOW2_DATASET_REPO_ID="${LAB01_FOLLOW2_DATASET_REPO_ID:-${LAB01_DATASET_HF_USER}/${LAB01_FOLLOW2_DATASET_NAME}}"

# Hub repos for this line. The 50-episode teleop repos stay in the template _env.sh.
LAB01_SMOLVLA_FOLLOW2_HF_REPO_ID="${LAB01_SMOLVLA_FOLLOW2_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-smolvla-agent-yaw-follow2}"
LAB01_SMOLVLA_FOLLOW2_POLICY_PATH="${LAB01_SMOLVLA_FOLLOW2_POLICY_PATH:-./outputs/train/lab01_pnp_smolvla_agent_yaw_v4_actonly_follow2/checkpoints/002500/pretrained_model}"
LAB01_MOLMO_AGENT193_HF_REPO_ID="${LAB01_MOLMO_AGENT193_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-molmoact2-agent193-20k}"
LAB01_MOLMO_AGENT193_POLICY_PATH="${LAB01_MOLMO_AGENT193_POLICY_PATH:-./outputs/train/lab01_pnp_molmoact2_state6_agent193_plus20k/checkpoints/020000/pretrained_model}"
LAB01_PI05_AGENT193_HF_REPO_ID="${LAB01_PI05_AGENT193_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-pi05-agent193}"
LAB01_PI05_AGENT193_POLICY_PATH="${LAB01_PI05_AGENT193_POLICY_PATH:-./outputs/train/lab01_pnp_pi05_state6_agent193/checkpoints/010000/pretrained_model}"
LAB01_PI05_UNFREEZE_HF_REPO_ID="${LAB01_PI05_UNFREEZE_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-pi05-agent193-unfreeze}"
LAB01_PI05_UNFREEZE_POLICY_PATH="${LAB01_PI05_UNFREEZE_POLICY_PATH:-/tmp/lab01_pi05_front_unfreezeA2_15k/checkpoints/005000/pretrained_model}"
LAB01_GROOT_AGENT193_ABS_HF_REPO_ID="${LAB01_GROOT_AGENT193_ABS_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-groot-n17-agent193-abs10k}"
# Push scripts still default to the historical /tmp checkpoints from the
# published run. train_dm05.cmd / train_xvla.cmd write under outputs/train/.
LAB01_DM05_AGENT193_HF_REPO_ID="${LAB01_DM05_AGENT193_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-dm05-agent193-10k}"
LAB01_DM05_AGENT193_POLICY_PATH="${LAB01_DM05_AGENT193_POLICY_PATH:-/tmp/lab01_dm05_20k/checkpoints/020000/pretrained_model}"
LAB01_XVLA_AGENT193_HF_REPO_ID="${LAB01_XVLA_AGENT193_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-xvla-agent193-10k}"
LAB01_XVLA_AGENT193_POLICY_PATH="${LAB01_XVLA_AGENT193_POLICY_PATH:-/tmp/lab01_xvla_10k/checkpoints/010000/pretrained_model}"
LAB01_GROOT_AGENT193_ABS_POLICY_PATH="${LAB01_GROOT_AGENT193_ABS_POLICY_PATH:-./outputs/train/lab01_pnp_groot_n17_state6_agent193_abs10k/checkpoints/010000/pretrained_model}"
# LingBot-VLA 2.0 (overrides stay in ../lab01_pnp/_env.sh section 5c; Hub id mirrored here for agent cards/push)
LAB01_V2_HF_REPO_ID="${LAB01_V2_HF_REPO_ID:-alexhegit/so101-simstudio-lab01-pnp-lingbot-vla-v2-agent193-20k}"
