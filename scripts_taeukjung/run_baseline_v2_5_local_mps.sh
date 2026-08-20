#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/baseline_v2_5_local}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-4}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-4}"
export SOLAR_D_MODEL="${SOLAR_D_MODEL:-128}"
export SOLAR_WIND_DIM="${SOLAR_WIND_DIM:-64}"
export SOLAR_NHEAD="${SOLAR_NHEAD:-4}"
export SOLAR_ENCODER_LAYERS="${SOLAR_ENCODER_LAYERS:-2}"
export SOLAR_FF_DIM="${SOLAR_FF_DIM:-320}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.15}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.20}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"
export V25_AR_ORDER="${V25_AR_ORDER:-2}"
export V25_AR_RIDGE="${V25_AR_RIDGE:-30}"
export V25_WIND_RESIDUAL_CAP_MULTIPLIER="${V25_WIND_RESIDUAL_CAP_MULTIPLIER:-1.0}"
export V25_FIXED_LAG_HOURS="${V25_FIXED_LAG_HOURS:-96}"
export V25_FIXED_LAG_SIGMA_HOURS="${V25_FIXED_LAG_SIGMA_HOURS:-12}"
export V25_FIXED_LAG_WINDOW_HOURS="${V25_FIXED_LAG_WINDOW_HOURS:-24}"
export V25_LOCAL_TEMPORAL_RADIUS_HOURS="${V25_LOCAL_TEMPORAL_RADIUS_HOURS:-12}"
export V25_CENTRAL_SPATIAL_PRIOR_STRENGTH="${V25_CENTRAL_SPATIAL_PRIOR_STRENGTH:-0.35}"
export V25_IMAGE_SCALE_LIMIT="${V25_IMAGE_SCALE_LIMIT:-0.30}"
export V25_IMAGE_SCALE_HIDDEN_DIM="${V25_IMAGE_SCALE_HIDDEN_DIM:-96}"
export V25_INITIAL_IMAGE_GATE="${V25_INITIAL_IMAGE_GATE:-0.40}"
export V25_IMAGE_HEAD_INIT_STD="${V25_IMAGE_HEAD_INIT_STD:-0.001}"
export V25_WIND_HEAD_INIT_STD="${V25_WIND_HEAD_INIT_STD:-0.0002}"
export V25_DELTA_GAIN="${V25_DELTA_GAIN:-4.0}"
export V25_TIME_MASK_PROBABILITY="${V25_TIME_MASK_PROBABILITY:-0.02}"
export V25_MODALITY_DROP_PROBABILITY="${V25_MODALITY_DROP_PROBABILITY:-0.10}"
export V25_IMAGE_RESIDUAL_CAP_MULTIPLIER="${V25_IMAGE_RESIDUAL_CAP_MULTIPLIER:-1.5}"
export V25_RESIDUAL_L2_WEIGHT="${V25_RESIDUAL_L2_WEIGHT:-0.0005}"
export V25_EMA_DECAY="${V25_EMA_DECAY:-0.99}"
export V25_WARMUP_EPOCHS="${V25_WARMUP_EPOCHS:-2}"
export V25_MIN_LR="${V25_MIN_LR:-2e-6}"
export V25_WEIGHT_DECAY="${V25_WEIGHT_DECAY:-0.01}"
export V25_EARLY_STOP_PATIENCE="${V25_EARLY_STOP_PATIENCE:-12}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ASAI" ]]; then
  echo "activate the ASAI environment first: conda activate ASAI" >&2
  exit 2
fi

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_baseline_v2_5.py
    ;;
  infer)
    exec python src_taeukjung/inference_baseline_v2_5.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_baseline_v2_5.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
