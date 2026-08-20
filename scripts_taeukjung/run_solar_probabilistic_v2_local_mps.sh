#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_probabilistic_v2_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-128}"
export EPOCHS="${EPOCHS:-20}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export IMAGE_NORM="${IMAGE_NORM:-soft_cubic}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.25}"
export SOLAR_V2_SPATIAL_HEIGHT="${SOLAR_V2_SPATIAL_HEIGHT:-4}"
export SOLAR_V2_SPATIAL_WIDTH="${SOLAR_V2_SPATIAL_WIDTH:-8}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.25}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export PROBABILISTIC_NLL_WEIGHT="${PROBABILISTIC_NLL_WEIGHT:-5.0}"

conda run --no-capture-output -n ASAI python -c \
  'import torch; assert torch.backends.mps.is_available(), "Apple MPS is unavailable"'

case "${ACTION}" in
  train)
    exec conda run --no-capture-output -n ASAI \
      python src_taeukjung/train_solar_probabilistic_v2.py
    ;;
  infer)
    exec conda run --no-capture-output -n ASAI \
      python src_taeukjung/inference_solar_probabilistic_v2.py
    ;;
  *)
    echo "usage: $0 [train|infer]" >&2
    exit 2
    ;;
esac
