#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-smoke}"

cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

if [[ "${CONDA_DEFAULT_ENV:-}" != "ASAI" ]]; then
  echo "activate the ASAI environment first: conda activate ASAI" >&2
  exit 2
fi

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  smoke)
    exec python src_taeukjung/smoke_solar_lstm_unet_v12_1.py
    ;;
  train)
    export SEED="${SEED:-777}"
    export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
    export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_lag_lstm_unet_v12_1_local_smoke}"
    export IMAGE_SIZE="${IMAGE_SIZE:-64}"
    export IMAGE_NORM="${IMAGE_NORM:-linear}"
    export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
    export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
    export EPOCHS="${EPOCHS:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-4}"
    export NUM_WORKERS="${NUM_WORKERS:-0}"
    export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-32}"
    export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-16}"
    export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
    export V12_1_UNET_CHANNELS="${V12_1_UNET_CHANNELS:-12,16,24,40,56}"
    export V12_LAG_HOURS="${V12_LAG_HOURS:-96}"
    export V12_LAG_PRIOR_MAX_STRENGTH="${V12_LAG_PRIOR_MAX_STRENGTH:-2.0}"
    export V12_LAG_ALIGNMENT_WEIGHT="${V12_LAG_ALIGNMENT_WEIGHT:-0.005}"
    export V12_TIME_MASK_PROBABILITY="${V12_TIME_MASK_PROBABILITY:-0.15}"
    export V12_MODALITY_DROP_PROBABILITY="${V12_MODALITY_DROP_PROBABILITY:-0.25}"
    export V12_IMAGE_CORRECTION_CAP_MULTIPLIER="${V12_IMAGE_CORRECTION_CAP_MULTIPLIER:-1.25}"
    export V12_CORRECTION_L2_WEIGHT="${V12_CORRECTION_L2_WEIGHT:-0.10}"
    exec python src_taeukjung/train_solar_lstm_unet_v12_1.py
    ;;
  *)
    echo "usage: $0 [smoke|train]" >&2
    exit 2
    ;;
esac
