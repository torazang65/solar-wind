#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/outputs/tile_transformer_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export EPOCHS="${EPOCHS:-60}"
export BATCH_SIZE="${BATCH_SIZE:-256}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export IMAGE_NORM="${IMAGE_NORM:-soft_cubic}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.25}"
export TILE_GRID_SIZE="${TILE_GRID_SIZE:-8}"
export TILE_D_MODEL="${TILE_D_MODEL:-128}"
export TILE_WIND_DIM="${TILE_WIND_DIM:-32}"
export TILE_NHEAD="${TILE_NHEAD:-8}"
export TILE_ENCODER_LAYERS="${TILE_ENCODER_LAYERS:-2}"
export TILE_FF_DIM="${TILE_FF_DIM:-256}"
export TILE_DROPOUT="${TILE_DROPOUT:-0.1}"
export LEARNING_RATE="${LEARNING_RATE:-3e-4}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_tile_transformer.py
    ;;
  infer)
    exec python src_taeukjung/inference_tile_transformer.py
    ;;
  *)
    echo "usage: $0 [train|infer]" >&2
    exit 2
    ;;
esac
