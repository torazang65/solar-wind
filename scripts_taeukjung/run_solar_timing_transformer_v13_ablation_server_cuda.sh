#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V13_ABLATION_ROOT="${V13_ABLATION_ROOT:-/home/jovyan/outputs/solar_timing_transformer_v13_seed${SEED}}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export EPOCHS="${EPOCHS:-25}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"

export V13_GRID_ROWS="${V13_GRID_ROWS:-2}"
export V13_GRID_COLUMNS="${V13_GRID_COLUMNS:-8}"
export V13_UNET_CHANNELS="${V13_UNET_CHANNELS:-12,16,24,40,56}"
export V13_D_MODEL="${V13_D_MODEL:-96}"
export V13_ATTENTION_HEADS="${V13_ATTENTION_HEADS:-4}"
export V13_DECODER_LAYERS="${V13_DECODER_LAYERS:-2}"
export V13_FEEDFORWARD_DIM="${V13_FEEDFORWARD_DIM:-192}"
export V13_DROPOUT="${V13_DROPOUT:-0.15}"
export V13_DELTA_GAIN="${V13_DELTA_GAIN:-1.0}"
export V13_TIME_MASK_PROBABILITY="${V13_TIME_MASK_PROBABILITY:-0.15}"
export V13_MODALITY_DROP_PROBABILITY="${V13_MODALITY_DROP_PROBABILITY:-0.25}"
export V13_TIMING_SIGMA_HOURS="${V13_TIMING_SIGMA_HOURS:-18}"
export V13_PHYSICAL_PRIOR_MIN="${V13_PHYSICAL_PRIOR_MIN:-1.0}"
export V13_PHYSICAL_PRIOR_MAX="${V13_PHYSICAL_PRIOR_MAX:-4.0}"
export V13_PHYSICAL_PRIOR_INIT="${V13_PHYSICAL_PRIOR_INIT:-2.0}"
export V13_MAXIMUM_BLEND="${V13_MAXIMUM_BLEND:-0.50}"
export V13_INITIAL_BLEND="${V13_INITIAL_BLEND:-0.05}"
export V13_CORRECTION_CAP_MULTIPLIER="${V13_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V13_HINDCAST_WEIGHT_START="${V13_HINDCAST_WEIGHT_START:-0.50}"
export V13_HINDCAST_WEIGHT_END="${V13_HINDCAST_WEIGHT_END:-0.10}"
export V13_HINDCAST_DECAY_EPOCHS="${V13_HINDCAST_DECAY_EPOCHS:-8}"
export V13_ALIGNMENT_SIGMA_DEG="${V13_ALIGNMENT_SIGMA_DEG:-12}"
export V13_CORRECTION_L2_WEIGHT="${V13_CORRECTION_L2_WEIGHT:-0.10}"
export V13_GATE_L1_WEIGHT="${V13_GATE_L1_WEIGHT:-0.01}"
export V13_SPEED_SMOOTHNESS_WEIGHT="${V13_SPEED_SMOOTHNESS_WEIGHT:-0.02}"
export V13_GRADIENT_CLIP="${V13_GRADIENT_CLIP:-1.0}"
export V13_WEIGHT_DECAY="${V13_WEIGHT_DECAY:-0.03}"
export V13_WARMUP_EPOCHS="${V13_WARMUP_EPOCHS:-3}"
export V13_MIN_LR="${V13_MIN_LR:-1e-6}"
export V13_EARLY_STOP_PATIENCE="${V13_EARLY_STOP_PATIENCE:-6}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "smoke" ]]; then
  exec python src_taeukjung/smoke_solar_timing_transformer_v13.py
fi

run_experiment() {
  local name="$1"
  local alignment_weight="$2"
  local output_dir="${V13_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_timing_transformer_v13.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_timing_transformer_v13.py"
      ;;
    *)
      echo "usage: $0 [smoke|train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: image=${IMAGE_SIZE} grid=${V13_GRID_ROWS}x${V13_GRID_COLUMNS} backmapping=${alignment_weight}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V13_ALIGNMENT_WEIGHT="${alignment_weight}" \
    python "${program}"
}

EXPERIMENTS="${V13_EXPERIMENTS:-v13_full_backmapping v13_no_backmapping}"

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/smoke_solar_timing_transformer_v13.py
fi

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    v13_full_backmapping)
      run_experiment "${experiment}" 0.01
      ;;
    v13_no_backmapping)
      run_experiment "${experiment}" 0.0
      ;;
    *)
      echo "unknown V13 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_timing_transformer_v13.py
fi
