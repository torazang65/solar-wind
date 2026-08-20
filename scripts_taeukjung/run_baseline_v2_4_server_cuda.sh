#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/outputs/baseline_v2_4_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-4}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-4}"
export SOLAR_D_MODEL="${SOLAR_D_MODEL:-128}"
export SOLAR_WIND_DIM="${SOLAR_WIND_DIM:-64}"
export SOLAR_NHEAD="${SOLAR_NHEAD:-4}"
export SOLAR_ENCODER_LAYERS="${SOLAR_ENCODER_LAYERS:-1}"
export SOLAR_FF_DIM="${SOLAR_FF_DIM:-256}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.20}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.25}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"
export V24_AR_ORDER="${V24_AR_ORDER:-2}"
export V24_AR_RIDGE="${V24_AR_RIDGE:-30}"
export V24_WIND_RESIDUAL_CAP_MULTIPLIER="${V24_WIND_RESIDUAL_CAP_MULTIPLIER:-0.75}"
export V24_FIXED_LAG_HOURS="${V24_FIXED_LAG_HOURS:-96}"
export V24_FIXED_LAG_SIGMA_HOURS="${V24_FIXED_LAG_SIGMA_HOURS:-12}"
export V24_FIXED_LAG_WINDOW_HOURS="${V24_FIXED_LAG_WINDOW_HOURS:-24}"
export V24_LOCAL_TEMPORAL_RADIUS_HOURS="${V24_LOCAL_TEMPORAL_RADIUS_HOURS:-12}"
export V24_CENTRAL_SPATIAL_PRIOR_STRENGTH="${V24_CENTRAL_SPATIAL_PRIOR_STRENGTH:-0.5}"
export V24_IMAGE_SCALE_LIMIT="${V24_IMAGE_SCALE_LIMIT:-0.15}"
export V24_DELTA_GAIN="${V24_DELTA_GAIN:-4.0}"
export V24_TIME_MASK_PROBABILITY="${V24_TIME_MASK_PROBABILITY:-0.05}"
export V24_MODALITY_DROP_PROBABILITY="${V24_MODALITY_DROP_PROBABILITY:-0.20}"
export V24_IMAGE_RESIDUAL_CAP_MULTIPLIER="${V24_IMAGE_RESIDUAL_CAP_MULTIPLIER:-1.0}"
export V24_RESIDUAL_L2_WEIGHT="${V24_RESIDUAL_L2_WEIGHT:-0.002}"
export V24_EMA_DECAY="${V24_EMA_DECAY:-0.995}"
export V24_WARMUP_EPOCHS="${V24_WARMUP_EPOCHS:-3}"
export V24_MIN_LR="${V24_MIN_LR:-1e-6}"
export V24_WEIGHT_DECAY="${V24_WEIGHT_DECAY:-0.02}"
export V24_EARLY_STOP_PATIENCE="${V24_EARLY_STOP_PATIENCE:-15}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_baseline_v2_4.py
    ;;
  infer)
    exec python src_taeukjung/inference_baseline_v2_4.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_baseline_v2_4.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
