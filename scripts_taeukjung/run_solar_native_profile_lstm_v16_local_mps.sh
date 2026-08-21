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
    exec python src_taeukjung/smoke_solar_native_profile_lstm_v16.py
    ;;
  train)
    export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/../dev/public_dataset/competition_dataset_6h}"
    export CACHE_DIR="${CACHE_DIR:-${REPO_ROOT}/../dev/outputs/cache_taeukjung}"
    export OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/../dev/outputs/solar_native_profile_lstm_v16_local_smoke}"
    export IMAGE_SIZE=64 IMAGE_NORM=linear SOFT_CUBIC_STRENGTH=0 SOLAR_DISK_MASK=1
    export EPOCHS="${EPOCHS:-1}" BATCH_SIZE="${BATCH_SIZE:-2}" NUM_WORKERS=0
    export MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-16}"
    export MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-8}"
    export V16_COLUMN_DIM="${V16_COLUMN_DIM:-12}"
    export V16_FRAME_DIM="${V16_FRAME_DIM:-64}"
    export V16_LSTM_HIDDEN_DIM="${V16_LSTM_HIDDEN_DIM:-48}"
    export V16_WIND_FEATURE_DIM="${V16_WIND_FEATURE_DIM:-32}"
    export V16_LAG_HOURS="${V16_LAG_HOURS:-96}"
    export V16_WIND_AUX_WEIGHT=0 V16_LAG_ALIGNMENT_WEIGHT=0
    exec python src_taeukjung/train_solar_native_profile_lstm_v16.py
    ;;
  *)
    echo "usage: $0 [smoke|train]" >&2
    exit 2
    ;;
esac
