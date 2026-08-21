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
    exec python src_taeukjung/smoke_solar_peak_event_v15.py
    ;;
  train)
    export SEED="${SEED:-777}"
    export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
    export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_peak_event_v15_local_smoke}"
    export IMAGE_SIZE="${IMAGE_SIZE:-64}"
    export IMAGE_NORM="${IMAGE_NORM:-linear}"
    export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
    export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
    export EPOCHS="${EPOCHS:-1}"
    export BATCH_SIZE="${BATCH_SIZE:-2}"
    export NUM_WORKERS="${NUM_WORKERS:-0}"
    export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16}"
    export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-8}"
    export LEARNING_RATE="${LEARNING_RATE:-3e-5}"
    export V15_GRID_ROWS="${V15_GRID_ROWS:-2}"
    export V15_GRID_COLUMNS="${V15_GRID_COLUMNS:-8}"
    export V15_UNET_CHANNELS="${V15_UNET_CHANNELS:-8,12,16,24,32}"
    export V15_D_MODEL="${V15_D_MODEL:-64}"
    export V15_ATTENTION_HEADS="${V15_ATTENTION_HEADS:-4}"
    export V15_DECODER_LAYERS="${V15_DECODER_LAYERS:-1}"
    export V15_FEEDFORWARD_DIM="${V15_FEEDFORWARD_DIM:-128}"
    export V15_PEAK_HIDDEN_DIM="${V15_PEAK_HIDDEN_DIM:-64}"
    export V15_ALIGNMENT_WEIGHT="${V15_ALIGNMENT_WEIGHT:-0.0}"
    export V15_PEAK_TIME_LOSS_WEIGHT="${V15_PEAK_TIME_LOSS_WEIGHT:-0.05}"
    export V15_PEAK_VALUE_LOSS_WEIGHT="${V15_PEAK_VALUE_LOSS_WEIGHT:-0.25}"
    exec python src_taeukjung/train_solar_peak_event_v15.py
    ;;
  *)
    echo "usage: $0 [smoke|train]" >&2
    exit 2
    ;;
esac
