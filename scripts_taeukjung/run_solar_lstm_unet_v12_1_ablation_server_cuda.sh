#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V12_1_ABLATION_ROOT="${V12_1_ABLATION_ROOT:-/home/jovyan/outputs/solar_lag_lstm_unet_v12_1_seed${SEED}}"
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

export V12_GRID_ROWS="${V12_GRID_ROWS:-2}"
export V12_GRID_COLUMNS="${V12_GRID_COLUMNS:-8}"
export V12_CELL_DIM="${V12_CELL_DIM:-48}"
export V12_FRAME_DIM="${V12_FRAME_DIM:-256}"
export V12_LSTM_HIDDEN_DIM="${V12_LSTM_HIDDEN_DIM:-192}"
export V12_LSTM_LAYERS="${V12_LSTM_LAYERS:-1}"
export V12_WIND_FEATURE_DIM="${V12_WIND_FEATURE_DIM:-128}"
export V12_1_UNET_CHANNELS="${V12_1_UNET_CHANNELS:-12,16,24,40,56}"
export V12_DROPOUT="${V12_DROPOUT:-0.15}"
export V12_DELTA_GAIN="${V12_DELTA_GAIN:-1.0}"
export V12_TIME_MASK_PROBABILITY="${V12_TIME_MASK_PROBABILITY:-0.15}"
export V12_MODALITY_DROP_PROBABILITY="${V12_MODALITY_DROP_PROBABILITY:-0.25}"
export V12_LAG_SIGMA_HOURS="${V12_LAG_SIGMA_HOURS:-12}"
export V12_LAG_PRIOR_INIT_STRENGTH="${V12_LAG_PRIOR_INIT_STRENGTH:-1.0}"
export V12_WIND_RESIDUAL_CAP_MULTIPLIER="${V12_WIND_RESIDUAL_CAP_MULTIPLIER:-1.0}"
export V12_IMAGE_CORRECTION_CAP_MULTIPLIER="${V12_IMAGE_CORRECTION_CAP_MULTIPLIER:-1.25}"
export V12_WIND_AUX_WEIGHT="${V12_WIND_AUX_WEIGHT:-0.20}"
export V12_ALIGNMENT_SIGMA_HOURS="${V12_ALIGNMENT_SIGMA_HOURS:-12}"
export V12_CORRECTION_L2_WEIGHT="${V12_CORRECTION_L2_WEIGHT:-0.10}"
export V12_GRADIENT_CLIP="${V12_GRADIENT_CLIP:-1.0}"
export V12_WEIGHT_DECAY="${V12_WEIGHT_DECAY:-0.03}"
export V12_WARMUP_EPOCHS="${V12_WARMUP_EPOCHS:-3}"
export V12_MIN_LR="${V12_MIN_LR:-1e-6}"
export V12_EARLY_STOP_PATIENCE="${V12_EARLY_STOP_PATIENCE:-6}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

run_experiment() {
  local name="$1"
  local lag_hours="$2"
  local prior_max="$3"
  local alignment_weight="$4"
  local output_dir="${V12_1_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_lstm_unet_v12_1.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_lstm_unet_v12_1.py"
      ;;
    *)
      echo "usage: $0 [train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: lag_hours=${lag_hours} prior_max=${prior_max} alignment=${alignment_weight}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V12_LAG_HOURS="${lag_hours}" \
    V12_LAG_PRIOR_MAX_STRENGTH="${prior_max}" \
    V12_LAG_ALIGNMENT_WEIGHT="${alignment_weight}" \
    python "${program}"
}

EXPERIMENTS="${V12_1_EXPERIMENTS:-unet_fixed96_guarded unet_multilag_guarded}"

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    unet_fixed96_guarded)
      run_experiment "${experiment}" "96" 2.0 0.005
      ;;
    unet_multilag_guarded)
      run_experiment "${experiment}" "72,84,96,108,120" 2.0 0.005
      ;;
    *)
      echo "unknown V12.1 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_lstm_unet_v12_1.py
fi
