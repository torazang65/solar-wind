#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_ballistic_v8_local}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export SOLAR_CEA_RADIUS_FRACTION="${SOLAR_CEA_RADIUS_FRACTION:-0.42}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-4}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-4}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.05}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.15}"
export SOLAR_VISUAL_DROPOUT="${SOLAR_VISUAL_DROPOUT:-0.10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export SOLAR_V8_PHYSICS_PRIOR_STRENGTH="${SOLAR_V8_PHYSICS_PRIOR_STRENGTH:-1.0}"
export SOLAR_V8_LONGITUDE_SIGMA_DEGREES="${SOLAR_V8_LONGITUDE_SIGMA_DEGREES:-30.0}"
export SOLAR_V8_LATITUDE_SIGMA_DEGREES="${SOLAR_V8_LATITUDE_SIGMA_DEGREES:-45.0}"
export SOLAR_V8_RESIDUAL_CAP_MULTIPLIER="${SOLAR_V8_RESIDUAL_CAP_MULTIPLIER:-2.5}"
export SOLAR_V8_NORTH_SOUTH_FLIP_PROBABILITY="${SOLAR_V8_NORTH_SOUTH_FLIP_PROBABILITY:-0.5}"
export SOLAR_V8_RESIDUAL_L2_WEIGHT="${SOLAR_V8_RESIDUAL_L2_WEIGHT:-0.01}"
export SOLAR_V8_EMA_DECAY="${SOLAR_V8_EMA_DECAY:-0.995}"

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_ballistic_v8.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_ballistic_v8.py
    ;;
  *)
    echo "usage: $0 [train|infer]" >&2
    exit 2
    ;;
esac
