#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V16_ABLATION_ROOT="${V16_ABLATION_ROOT:-/home/jovyan/outputs/solar_native_profile_lstm_v16_seed${SEED}}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export EPOCHS="${EPOCHS:-25}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"

export V16_COLUMN_DIM="${V16_COLUMN_DIM:-16}"
export V16_LONGITUDE_KERNEL_SIZE="${V16_LONGITUDE_KERNEL_SIZE:-5}"
export V16_FRAME_DIM="${V16_FRAME_DIM:-128}"
export V16_LSTM_HIDDEN_DIM="${V16_LSTM_HIDDEN_DIM:-96}"
export V16_LSTM_LAYERS="${V16_LSTM_LAYERS:-1}"
export V16_WIND_FEATURE_DIM="${V16_WIND_FEATURE_DIM:-64}"
export V16_DROPOUT="${V16_DROPOUT:-0.15}"
export V16_TIME_MASK_PROBABILITY="${V16_TIME_MASK_PROBABILITY:-0.15}"
export V16_MODALITY_DROP_PROBABILITY="${V16_MODALITY_DROP_PROBABILITY:-0.25}"
export V16_DELTA_GAIN="${V16_DELTA_GAIN:-1.0}"
export V16_LAG_HOURS="${V16_LAG_HOURS:-96}"
export V16_LAG_SIGMA_HOURS="${V16_LAG_SIGMA_HOURS:-12}"
export V16_LAG_PRIOR_MAX_STRENGTH="${V16_LAG_PRIOR_MAX_STRENGTH:-2.0}"
export V16_LAG_PRIOR_INIT_STRENGTH="${V16_LAG_PRIOR_INIT_STRENGTH:-1.0}"
export V16_DISABLE_WIND_RESIDUAL="${V16_DISABLE_WIND_RESIDUAL:-1}"
export V16_IMAGE_CORRECTION_CAP_MULTIPLIER="${V16_IMAGE_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V16_WIND_AUX_WEIGHT="${V16_WIND_AUX_WEIGHT:-0.0}"
export V16_LAG_ALIGNMENT_WEIGHT="${V16_LAG_ALIGNMENT_WEIGHT:-0.0}"
export V16_CORRECTION_L2_WEIGHT="${V16_CORRECTION_L2_WEIGHT:-0.05}"
export V16_WEIGHT_DECAY="${V16_WEIGHT_DECAY:-0.03}"
export V16_WARMUP_EPOCHS="${V16_WARMUP_EPOCHS:-3}"
export V16_MIN_LR="${V16_MIN_LR:-1e-6}"
export V16_GRADIENT_CLIP="${V16_GRADIENT_CLIP:-1.0}"
export V16_EARLY_STOP_PATIENCE="${V16_EARLY_STOP_PATIENCE:-6}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "smoke" ]]; then
  exec python src_taeukjung/smoke_solar_native_profile_lstm_v16.py
fi

run_experiment() {
  local name="$1"
  local scramble_images="$2"
  local wind_only="$3"
  local output_dir="${V16_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_native_profile_lstm_v16.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_native_profile_lstm_v16.py"
      ;;
    *)
      echo "usage: $0 [smoke|train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: native_columns=${IMAGE_SIZE} scramble=${scramble_images} wind_only=${wind_only}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V16_SCRAMBLE_IMAGES="${scramble_images}" \
    WIND_ONLY="${wind_only}" \
    python "${program}"
}

EXPERIMENTS="${V16_EXPERIMENTS:-v16_native v16_scrambled v16_wind_only}"

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/smoke_solar_native_profile_lstm_v16.py
fi

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    v16_native)
      run_experiment "${experiment}" 0 0
      ;;
    v16_scrambled)
      run_experiment "${experiment}" 1 0
      ;;
    v16_wind_only)
      run_experiment "${experiment}" 0 1
      ;;
    *)
      echo "unknown V16 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_native_profile_lstm_v16.py
fi
