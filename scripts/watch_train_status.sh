#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="${1:-${SCRIPT_DIR}/diffusion_policy_official_base.log}"
INTERVAL="${INTERVAL:-2}"

if [[ ! -f "${LOG_FILE}" ]]; then
  echo "Log file not found: ${LOG_FILE}"
  exit 1
fi

while true; do
  clear
  date
  echo
  echo "Log: ${LOG_FILE}"
  echo
  echo "Latest training metrics:"
  grep 'step:' "${LOG_FILE}" | tail -n 5 || true
  echo
  echo "Checkpoint / eval events:"
  grep -E 'Saving checkpoint|checkpoint saved|Eval|evaluation' "${LOG_FILE}" | tail -n 5 || true
  echo
  echo "GPU:"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo
  echo "Press Ctrl-C to stop. Refresh interval: ${INTERVAL}s"
  sleep "${INTERVAL}"
done
