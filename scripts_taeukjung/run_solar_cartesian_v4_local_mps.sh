#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_cartesian_v4_local_64}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-12}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOLAR_V3_SPATIAL_HEIGHT="${SOLAR_V3_SPATIAL_HEIGHT:-4}"
export SOLAR_V3_SPATIAL_WIDTH="${SOLAR_V3_SPATIAL_WIDTH:-8}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.25}"
export SOLAR_VISUAL_DROPOUT="${SOLAR_VISUAL_DROPOUT:-0.10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"

conda run --no-capture-output -n ASAI python -c \
  'import torch; assert torch.backends.mps.is_available(), "Apple MPS is unavailable"'

exec conda run --no-capture-output -n ASAI \
  python src_taeukjung/train_solar_cartesian_v4.py
