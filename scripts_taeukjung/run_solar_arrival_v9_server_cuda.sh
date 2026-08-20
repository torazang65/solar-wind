#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/outputs/solar_arrival_v9_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-128}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export SOLAR_CEA_RADIUS_FRACTION="${SOLAR_CEA_RADIUS_FRACTION:-0.42}"
export IMAGE_NORM="${IMAGE_NORM:-soft_cubic}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.25}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-2}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-4}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.05}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.15}"
export SOLAR_VISUAL_DROPOUT="${SOLAR_VISUAL_DROPOUT:-0.10}"
export LEARNING_RATE="${LEARNING_RATE:-2e-4}"
export SOLAR_V9_IMAGE_CNN_CHANNELS="${SOLAR_V9_IMAGE_CNN_CHANNELS:-48}"
export SOLAR_V9_TEMPORAL_LAYERS="${SOLAR_V9_TEMPORAL_LAYERS:-3}"
export SOLAR_V9_TIME_MASK_PROBABILITY="${SOLAR_V9_TIME_MASK_PROBABILITY:-0.10}"
export SOLAR_V9_MODALITY_DROP_PROBABILITY="${SOLAR_V9_MODALITY_DROP_PROBABILITY:-0.10}"
export SOLAR_V9_TRANSIT_MIN_HOURS="${SOLAR_V9_TRANSIT_MIN_HOURS:-48}"
export SOLAR_V9_TRANSIT_MAX_HOURS="${SOLAR_V9_TRANSIT_MAX_HOURS:-120}"
export SOLAR_V9_ARRIVAL_SIGMA_HOURS="${SOLAR_V9_ARRIVAL_SIGMA_HOURS:-24}"
export SOLAR_V9_ARRIVAL_PRIOR_STRENGTH="${SOLAR_V9_ARRIVAL_PRIOR_STRENGTH:-0.25}"
export SOLAR_V9_RESIDUAL_CAP_MULTIPLIER="${SOLAR_V9_RESIDUAL_CAP_MULTIPLIER:-2.5}"
export SOLAR_V9_RESIDUAL_L2_WEIGHT="${SOLAR_V9_RESIDUAL_L2_WEIGHT:-0.002}"
export SOLAR_V9_EMA_DECAY="${SOLAR_V9_EMA_DECAY:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_arrival_v9.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_arrival_v9.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_solar_arrival_v9.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
