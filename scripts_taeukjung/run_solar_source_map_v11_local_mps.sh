#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_source_map_v11_local_seed${SEED}}"
export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

export V11_AR_ORDER="${V11_AR_ORDER:-2}"
export V11_AR_RIDGE="${V11_AR_RIDGE:-30}"
export V11_SOURCE_HIDDEN_DIM="${V11_SOURCE_HIDDEN_DIM:-64}"
export V11_DROPOUT="${V11_DROPOUT:-0.10}"
export V11_FIXED_LAG_HOURS="${V11_FIXED_LAG_HOURS:-96}"
export V11_FIXED_LAG_REFERENCE_SPEED_KMS="${V11_FIXED_LAG_REFERENCE_SPEED_KMS:-430}"
export V11_DELTA_GAIN="${V11_DELTA_GAIN:-4.0}"
export V11_TIME_MASK_PROBABILITY="${V11_TIME_MASK_PROBABILITY:-0.05}"
export V11_MODALITY_DROP_PROBABILITY="${V11_MODALITY_DROP_PROBABILITY:-0.15}"
export V11_PROPAGATION_CAP_MULTIPLIER="${V11_PROPAGATION_CAP_MULTIPLIER:-1.25}"
export V11_KERNEL_SIGMA_HOURS="${V11_KERNEL_SIGMA_HOURS:-12}"
export V11_TRANSIT_RESIDUAL_HOURS="${V11_TRANSIT_RESIDUAL_HOURS:-18}"
export V11_FAST_WIND_THRESHOLD_KMS="${V11_FAST_WIND_THRESHOLD_KMS:-550}"
export V11_FAST_WIND_SCALE_KMS="${V11_FAST_WIND_SCALE_KMS:-50}"
export V11_FAST_QUIET_SUPPRESSION="${V11_FAST_QUIET_SUPPRESSION:-0.50}"
export V11_HINDCAST_WEIGHT_START="${V11_HINDCAST_WEIGHT_START:-0.70}"
export V11_HINDCAST_WEIGHT_END="${V11_HINDCAST_WEIGHT_END:-0.10}"
export V11_HINDCAST_DECAY_EPOCHS="${V11_HINDCAST_DECAY_EPOCHS:-8}"
export V11_ALIGNMENT_WEIGHT="${V11_ALIGNMENT_WEIGHT:-0.02}"
export V11_ALIGNMENT_SIGMA_DEG="${V11_ALIGNMENT_SIGMA_DEG:-20}"
export V11_TRANSIT_RESIDUAL_L2="${V11_TRANSIT_RESIDUAL_L2:-0.003}"
export V11_SURGE_WEIGHT="${V11_SURGE_WEIGHT:-0.02}"
export V11_COMPONENT_L2="${V11_COMPONENT_L2:-0.001}"
export V11_EMA_DECAY="${V11_EMA_DECAY:-0.995}"
export V11_PHYSICAL_LR_MULT="${V11_PHYSICAL_LR_MULT:-10}"
export V11_WARMUP_EPOCHS="${V11_WARMUP_EPOCHS:-2}"
export V11_COSINE_DECAY_EPOCHS="${V11_COSINE_DECAY_EPOCHS:-30}"
export V11_MIN_LR="${V11_MIN_LR:-1e-6}"
export V11_WEIGHT_DECAY="${V11_WEIGHT_DECAY:-0.02}"
export V11_EARLY_STOP_PATIENCE="${V11_EARLY_STOP_PATIENCE:-10}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ASAI" ]]; then
  echo "activate the ASAI environment first: conda activate ASAI" >&2
  exit 2
fi

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_source_map_v11.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_source_map_v11.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_solar_source_map_v11.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
