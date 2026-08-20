#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-/home/jovyan/outputs/solar_factorized_v7_taeukjung}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export IMAGE_SIZE="${IMAGE_SIZE:-128}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export SOLAR_CEA_RADIUS_FRACTION="${SOLAR_CEA_RADIUS_FRACTION:-0.42}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOLAR_V5_LATITUDE_BINS="${SOLAR_V5_LATITUDE_BINS:-4}"
export SOLAR_V5_LONGITUDE_BINS="${SOLAR_V5_LONGITUDE_BINS:-8}"
export SOLAR_V5_WIND_AUX_WEIGHT="${SOLAR_V5_WIND_AUX_WEIGHT:-0.20}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-1}"
export SOLAR_DROPOUT="${SOLAR_DROPOUT:-0.25}"
export SOLAR_VISUAL_DROPOUT="${SOLAR_VISUAL_DROPOUT:-0.10}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_factorized_v7.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_factorized_v7.py
    ;;
  *)
    echo "usage: $0 [train|infer]" >&2
    exit 2
    ;;
esac
