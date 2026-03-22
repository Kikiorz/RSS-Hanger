#!/usr/bin/env bash
set -euo pipefail

# ============== [DIFFUSION TRAINING] ==============
# Diffusion Policy training script
# Default config: use_base=true, use_torque=false
# ==================================================

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Training configuration
USE_BASE="${USE_BASE:-true}"
USE_TORQUE="${USE_TORQUE:-false}"
STEPS="${STEPS:-40000}"
SAVE_FREQ="${SAVE_FREQ:-20000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
DEVICE="${DEVICE:-cuda}"

# Dataset configuration (using relative paths)
DATASET_NAME="${DATASET_NAME:-ACT-100-WHOLE-V30-fixed}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/${DATASET_NAME}}"

VARIANT_SUFFIX="base"
if [ "${USE_TORQUE}" = "true" ]; then
  VARIANT_SUFFIX="${VARIANT_SUFFIX}-torque"
fi

RUN_NAME="DP-${DATASET_NAME}-${VARIANT_SUFFIX}"
if [ -n "${RUN_TAG:-}" ]; then
  RUN_NAME="${RUN_NAME}-${RUN_TAG}"
fi

# Output configuration (using relative paths)
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/models/${RUN_NAME}}"
JOB_NAME="${JOB_NAME:-${RUN_NAME}}"
POLICY_REPO_ID="${POLICY_REPO_ID:-local/${RUN_NAME}}"
WANDB_PROJECT="${WANDB_PROJECT:-DP-${DATASET_NAME}}"
LOG_FILE="${LOG_FILE:-${SCRIPT_DIR}/${RUN_NAME}.log}"

echo ""
echo "========================================"
echo "Training ${JOB_NAME}"
echo "  policy: diffusion"
echo "  use_base: ${USE_BASE}"
echo "  use_torque: ${USE_TORQUE}"
echo "  steps: ${STEPS}"
echo "  save_freq: ${SAVE_FREQ}"
echo "  eval_freq: ${EVAL_FREQ}"
echo "  batch_size: ${BATCH_SIZE}"
echo "  Dataset: ${DATASET_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "========================================"
echo ""

# Clear log file
echo "Diffusion Training Log - $(date)" > "${LOG_FILE}"

# Train diffusion policy (single GPU)
python "${PROJECT_ROOT}/lerobot/src/lerobot/scripts/lerobot_train.py" \
  --policy.type diffusion \
  --dataset.repo_id "${DATASET_NAME}" \
  --dataset.root "${DATASET_ROOT}" \
  --dataset.video_backend pyav \
  --output_dir "${OUTPUT_DIR}" \
  --job_name "${JOB_NAME}" \
  --policy.device "${DEVICE}" \
  --policy.use_base "${USE_BASE}" \
  --policy.use_torque "${USE_TORQUE}" \
  --policy.repo_id "${POLICY_REPO_ID}" \
  --policy.push_to_hub false \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --log_freq "${LOG_FREQ}" \
  --eval_freq "${EVAL_FREQ}" \
  --save_freq "${SAVE_FREQ}" \
  --wandb.enable "${WANDB_ENABLE}" \
  --wandb.project "${WANDB_PROJECT}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training completed!"
echo "  Model: ${OUTPUT_DIR}"
echo "========================================"
