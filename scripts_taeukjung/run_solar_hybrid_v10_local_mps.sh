#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_hybrid_v10_local_seed${SEED}}"
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

export V10_D_MODEL="${V10_D_MODEL:-128}"
export V10_NHEAD="${V10_NHEAD:-8}"
export V10_ENCODER_LAYERS="${V10_ENCODER_LAYERS:-2}"
export V10_DECODER_LAYERS="${V10_DECODER_LAYERS:-1}"
export V10_FF_DIM="${V10_FF_DIM:-256}"
export V10_DROPOUT="${V10_DROPOUT:-0.15}"
export V10_AR_ORDER="${V10_AR_ORDER:-2}"
export V10_AR_RIDGE="${V10_AR_RIDGE:-30}"
export V10_FIXED_LAG_HOURS="${V10_FIXED_LAG_HOURS:-96}"
export V10_FIXED_LAG_REFERENCE_SPEED_KMS="${V10_FIXED_LAG_REFERENCE_SPEED_KMS:-430}"
export V10_DELTA_GAIN="${V10_DELTA_GAIN:-4.0}"
export V10_TIME_MASK_PROBABILITY="${V10_TIME_MASK_PROBABILITY:-0.05}"
export V10_MODALITY_DROP_PROBABILITY="${V10_MODALITY_DROP_PROBABILITY:-0.20}"
export V10_CORRECTION_DROP_PROBABILITY="${V10_CORRECTION_DROP_PROBABILITY:-0.30}"
export V10_WIND_RESIDUAL_CAP_MULTIPLIER="${V10_WIND_RESIDUAL_CAP_MULTIPLIER:-1.0}"
export V10_PROPAGATION_CAP_MULTIPLIER="${V10_PROPAGATION_CAP_MULTIPLIER:-1.25}"
export V10_CORRECTION_CAP_MULTIPLIER="${V10_CORRECTION_CAP_MULTIPLIER:-0.75}"
export V10_WIND_AUX_WEIGHT="${V10_WIND_AUX_WEIGHT:-0.15}"
export V10_HINDCAST_WEIGHT_START="${V10_HINDCAST_WEIGHT_START:-0.50}"
export V10_HINDCAST_WEIGHT_END="${V10_HINDCAST_WEIGHT_END:-0.15}"
export V10_TRANSIT_RESIDUAL_L2="${V10_TRANSIT_RESIDUAL_L2:-0.003}"
export V10_COMPONENT_L2="${V10_COMPONENT_L2:-0.001}"
export V10_EMA_DECAY="${V10_EMA_DECAY:-0.995}"
export V10_PHYSICAL_LR_MULT="${V10_PHYSICAL_LR_MULT:-20}"
export V10_WARMUP_EPOCHS="${V10_WARMUP_EPOCHS:-3}"
export V10_MIN_LR="${V10_MIN_LR:-1e-6}"
export V10_WEIGHT_DECAY="${V10_WEIGHT_DECAY:-0.02}"
export V10_EARLY_STOP_PATIENCE="${V10_EARLY_STOP_PATIENCE:-15}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "ASAI" ]]; then
  echo "activate the ASAI environment first: conda activate ASAI" >&2
  exit 2
fi

python -c 'import torch; assert torch.backends.mps.is_available(), "MPS is unavailable"; print("Apple MPS")'

case "${ACTION}" in
  train)
    exec python src_taeukjung/train_solar_hybrid_v10.py
    ;;
  infer)
    exec python src_taeukjung/inference_solar_hybrid_v10.py
    ;;
  diagnose)
    exec python src_taeukjung/diagnose_solar_hybrid_v10.py
    ;;
  *)
    echo "usage: $0 [train|infer|diagnose]" >&2
    exit 2
    ;;
esac
