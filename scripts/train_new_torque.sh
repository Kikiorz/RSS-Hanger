#!/usr/bin/env bash
set -euo pipefail
export PYTHONWARNINGS="ignore"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
export PYTHONPATH="${PROJECT_ROOT}/lerobot/src${PYTHONPATH:+:${PYTHONPATH}}"

DATASET_NAME="${DATASET_NAME:-ACT-WHOLE-DP-3CAM-BASE-TORQUE}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/${DATASET_NAME}}"
DEVICE="${DEVICE:-cuda}"

# Runtime settings
BATCH_SIZE="${BATCH_SIZE:-32}"
STEPS="${STEPS:-200000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-16}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"

RUN_NAME="${RUN_NAME:-diffusion_policy_official_base_torque}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/models/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-DP-${DATASET_NAME}}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/${RUN_NAME}.log}"

# Policy settings: keep close to official Diffusion defaults.
# Task-specific overrides: use base velocity and torque observations.
HORIZON="${HORIZON:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
N_OBS_STEPS="${N_OBS_STEPS:-2}"
BETA_SCHEDULE="${BETA_SCHEDULE:-squaredcos_cap_v2}"
CLIP_SAMPLE="${CLIP_SAMPLE:-true}"
CLIP_SAMPLE_RANGE="${CLIP_SAMPLE_RANGE:-2.0}"
PREDICTION_TYPE="${PREDICTION_TYPE:-epsilon}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-5000}"
USE_BASE="${USE_BASE:-true}"
USE_TORQUE="${USE_TORQUE:-true}"

printf '\n'
echo "========================================"
echo "Training ${JOB_NAME}"
echo "  policy: diffusion"
echo "  dataset: ${DATASET_ROOT}"
echo "  output: ${OUTPUT_DIR}"
echo "  device: ${DEVICE}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  steps: ${STEPS}"
echo "  save_freq: ${SAVE_FREQ}"
echo "  eval_freq: ${EVAL_FREQ}"
echo "  horizon: ${HORIZON}"
echo "  n_action_steps: ${N_ACTION_STEPS}"
echo "  n_obs_steps: ${N_OBS_STEPS}"
echo "  use_base: ${USE_BASE}"
echo "  use_torque: ${USE_TORQUE}"
echo "  beta_schedule: ${BETA_SCHEDULE}"
echo "  clip_sample: ${CLIP_SAMPLE}"
echo "  clip_sample_range: ${CLIP_SAMPLE_RANGE}"
echo "  prediction_type: ${PREDICTION_TYPE}"
echo "  optimizer_lr: ${OPTIMIZER_LR}"
echo "  scheduler_name: ${SCHEDULER_NAME}"
echo "  scheduler_warmup_steps: ${SCHEDULER_WARMUP_STEPS}"
echo "  log: ${LOG_FILE}"
echo "========================================"
printf '\n'

echo "Diffusion Training Log - $(date)" > "${LOG_FILE}"

CMD=(
  python "${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py"
  --policy.type diffusion
  --dataset.repo_id "${DATASET_NAME}"
  --dataset.root "${DATASET_ROOT}"
  --dataset.video_backend pyav
  --batch_size "${BATCH_SIZE}"
  --steps "${STEPS}"
  --output_dir "${OUTPUT_DIR}"
  --job_name "${JOB_NAME}"
  --policy.device "${DEVICE}"
  --policy.horizon "${HORIZON}"
  --policy.n_action_steps "${N_ACTION_STEPS}"
  --policy.n_obs_steps "${N_OBS_STEPS}"
  --policy.beta_schedule "${BETA_SCHEDULE}"
  --policy.clip_sample "${CLIP_SAMPLE}"
  --policy.clip_sample_range "${CLIP_SAMPLE_RANGE}"
  --policy.prediction_type "${PREDICTION_TYPE}"
  --policy.optimizer_lr "${OPTIMIZER_LR}"
  --policy.scheduler_name "${SCHEDULER_NAME}"
  --policy.scheduler_warmup_steps "${SCHEDULER_WARMUP_STEPS}"
  --policy.use_base "${USE_BASE}"
  --policy.use_torque "${USE_TORQUE}"
  --policy.push_to_hub false
  --log_freq "${LOG_FREQ}"
  --eval_freq "${EVAL_FREQ}"
  --save_freq "${SAVE_FREQ}"
  --num_workers "${NUM_WORKERS}"
  --wandb.enable "${WANDB_ENABLE}"
  --wandb.project "${WANDB_PROJECT}"
)

{
  "${CMD[@]}" "$@"
} 2>&1 | tee -a "${LOG_FILE}"

printf '\n'
echo "========================================"
echo "Training completed!"
echo "  Model: ${OUTPUT_DIR}"
echo "========================================"
