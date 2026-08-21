#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_source_map_v11_1_local_seed${SEED}}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export EPOCHS="${EPOCHS:-35}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"

export V111_D_MODEL="${V111_D_MODEL:-128}"
export V111_DROPOUT="${V111_DROPOUT:-0.10}"
export V111_DELTA_GAIN="${V111_DELTA_GAIN:-1.0}"
export V111_TIME_MASK_PROBABILITY="${V111_TIME_MASK_PROBABILITY:-0.15}"
export V111_MODALITY_DROP_PROBABILITY="${V111_MODALITY_DROP_PROBABILITY:-0.25}"
export V111_KERNEL_SIGMA_HOURS="${V111_KERNEL_SIGMA_HOURS:-12}"
export V111_TRANSIT_RESIDUAL_HOURS="${V111_TRANSIT_RESIDUAL_HOURS:-24}"
export V111_HINDCAST_WEIGHT_START="${V111_HINDCAST_WEIGHT_START:-0.70}"
export V111_HINDCAST_WEIGHT_END="${V111_HINDCAST_WEIGHT_END:-0.10}"
export V111_HINDCAST_DECAY_EPOCHS="${V111_HINDCAST_DECAY_EPOCHS:-8}"
export V111_ALIGNMENT_WEIGHT="${V111_ALIGNMENT_WEIGHT:-0.02}"
export V111_ALIGNMENT_SIGMA_DEG="${V111_ALIGNMENT_SIGMA_DEG:-20}"
export V111_TRANSIT_RESIDUAL_L2="${V111_TRANSIT_RESIDUAL_L2:-0.003}"
export V111_SURGE_WEIGHT="${V111_SURGE_WEIGHT:-0.02}"
export V111_PHYSICAL_LR_MULT="${V111_PHYSICAL_LR_MULT:-100}"
export V111_WARMUP_EPOCHS="${V111_WARMUP_EPOCHS:-3}"
export V111_MIN_LR="${V111_MIN_LR:-1e-6}"
export V111_WEIGHT_DECAY="${V111_WEIGHT_DECAY:-0.01}"
export V111_EARLY_STOP_PATIENCE="${V111_EARLY_STOP_PATIENCE:-15}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ASAI" ]]; then
  echo "activate the ASAI environment first: conda activate ASAI" >&2
  exit 2
fi

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_source_map_v11_1.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_source_map_v11_1.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_solar_source_map_v11_1.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
