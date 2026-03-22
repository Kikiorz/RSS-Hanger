#!/usr/bin/env bash
set -euo pipefail

# ============== [DIFFUSION TRAINING - TORQUE] ==============
# Whole-body Diffusion Policy training script with torque enabled.
# Default config targets the ACT-100-WHOLE-V30-fixed dataset and uses:
# - pretrained resnet18 backbone
# - base velocity conditioning
# - joint effort conditioning
# - batch size 64
# - 200k steps
# ===========================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

USE_BASE="${USE_BASE:-true}"
USE_TORQUE="${USE_TORQUE:-true}"
STEPS="${STEPS:-200000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
LOG_FREQ="${LOG_FREQ:-100}"
EVAL_FREQ="${EVAL_FREQ:-0}"
BATCH_SIZE="${BATCH_SIZE:-64}"
WANDB_ENABLE="${WANDB_ENABLE:-false}"
DEVICE="${DEVICE:-cuda}"
PRETRAINED_BACKBONE_WEIGHTS="${PRETRAINED_BACKBONE_WEIGHTS:-ResNet18_Weights.IMAGENET1K_V1}"
USE_GROUP_NORM="${USE_GROUP_NORM:-false}"
RESIZE_SHAPE="${RESIZE_SHAPE:-224,224}"
CROP_RATIO="${CROP_RATIO:-1.0}"
COMPILE_MODEL="${COMPILE_MODEL:-false}"
NUM_WORKERS="${NUM_WORKERS:-16}"
USE_AMP="${USE_AMP:-true}"
OPTIMIZER_LR="${OPTIMIZER_LR:-8e-5}"
SCHEDULER_NAME="${SCHEDULER_NAME:-cosine}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-2000}"

DATASET_NAME="${DATASET_NAME:-ACT-100-WHOLE-V30-fixed}"
DATASET_ROOT="${DATASET_ROOT:-${PROJECT_ROOT}/data/${DATASET_NAME}}"

VARIANT_SUFFIX="base-torque"
RUN_NAME="DP-${DATASET_NAME}-${VARIANT_SUFFIX}"
if [ -n "${RUN_TAG:-}" ]; then
  RUN_NAME="${RUN_NAME}-${RUN_TAG}"
fi

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
echo "  num_workers: ${NUM_WORKERS}"
echo "  optimizer_lr: ${OPTIMIZER_LR}"
echo "  scheduler_name: ${SCHEDULER_NAME}"
echo "  scheduler_warmup_steps: ${SCHEDULER_WARMUP_STEPS}"
echo "  pretrained_backbone_weights: ${PRETRAINED_BACKBONE_WEIGHTS}"
echo "  use_group_norm: ${USE_GROUP_NORM}"
echo "  resize_shape: ${RESIZE_SHAPE}"
echo "  crop_ratio: ${CROP_RATIO}"
echo "  compile_model: ${COMPILE_MODEL}"
echo "  Dataset: ${DATASET_ROOT}"
echo "  Output: ${OUTPUT_DIR}"
echo "  Log: ${LOG_FILE}"
echo "========================================"
echo ""

echo "Diffusion Torque Training Log - $(date)" > "${LOG_FILE}"

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
  --policy.pretrained_backbone_weights "${PRETRAINED_BACKBONE_WEIGHTS}" \
  --policy.use_group_norm "${USE_GROUP_NORM}" \
  --policy.resize_shape "[${RESIZE_SHAPE}]" \
  --policy.crop_ratio "${CROP_RATIO}" \
  --policy.compile_model "${COMPILE_MODEL}" \
  --policy.repo_id "${POLICY_REPO_ID}" \
  --policy.push_to_hub false \
  --batch_size "${BATCH_SIZE}" \
  --steps "${STEPS}" \
  --log_freq "${LOG_FREQ}" \
  --eval_freq "${EVAL_FREQ}" \
  --save_freq "${SAVE_FREQ}" \
  --num_workers "${NUM_WORKERS}" \
  --policy.use_amp "${USE_AMP}" \
  --policy.optimizer_lr "${OPTIMIZER_LR}" \
  --policy.scheduler_name "${SCHEDULER_NAME}" \
  --policy.scheduler_warmup_steps "${SCHEDULER_WARMUP_STEPS}" \
  --wandb.enable "${WANDB_ENABLE}" \
  --wandb.project "${WANDB_PROJECT}" \
  "$@" 2>&1 | tee -a "${LOG_FILE}"

echo ""
echo "========================================"
echo "Training completed!"
echo "  Model: ${OUTPUT_DIR}"
echo "========================================"
