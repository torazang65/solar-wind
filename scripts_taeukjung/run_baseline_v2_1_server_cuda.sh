#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/outputs/baseline_v2_1_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-80}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-4}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-4}"
export SOLAR_D_MODEL="${SOLAR_D_MODEL:-96}"
export SOLAR_WIND_DIM="${SOLAR_WIND_DIM:-64}"
export SOLAR_NHEAD="${SOLAR_NHEAD:-4}"
export SOLAR_ENCODER_LAYERS="${SOLAR_ENCODER_LAYERS:-1}"
export SOLAR_FF_DIM="${SOLAR_FF_DIM:-192}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.20}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.25}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"
export V21_DELTA_GAIN="${V21_DELTA_GAIN:-4.0}"
export V21_TIME_MASK_PROBABILITY="${V21_TIME_MASK_PROBABILITY:-0.15}"
export V21_MODALITY_DROP_PROBABILITY="${V21_MODALITY_DROP_PROBABILITY:-0.25}"
export V21_TIMING_PRIOR_STRENGTH="${V21_TIMING_PRIOR_STRENGTH:-0.10}"
export V21_TIMING_PRIOR_SIGMA_HOURS="${V21_TIMING_PRIOR_SIGMA_HOURS:-36}"
export V21_RESIDUAL_CAP_MULTIPLIER="${V21_RESIDUAL_CAP_MULTIPLIER:-1.5}"
export V21_RESIDUAL_L2_WEIGHT="${V21_RESIDUAL_L2_WEIGHT:-0.002}"
export V21_EMA_DECAY="${V21_EMA_DECAY:-0.995}"
export V21_WARMUP_EPOCHS="${V21_WARMUP_EPOCHS:-3}"
export V21_MIN_LR="${V21_MIN_LR:-1e-6}"
export V21_WEIGHT_DECAY="${V21_WEIGHT_DECAY:-0.02}"
export V21_EARLY_STOP_PATIENCE="${V21_EARLY_STOP_PATIENCE:-20}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_baseline_v2_1.py
    ;;
  infer)
    exec python src_taeukjung/inference_baseline_v2_1.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_baseline_v2_1.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
