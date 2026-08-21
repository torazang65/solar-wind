#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V14_ABLATION_ROOT="${V14_ABLATION_ROOT:-/home/jovyan/outputs/solar_deformable_timing_v14_seed${SEED}}"
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

export V14_GRID_ROWS="${V14_GRID_ROWS:-2}"
export V14_GRID_COLUMNS="${V14_GRID_COLUMNS:-8}"
export V14_UNET_CHANNELS="${V14_UNET_CHANNELS:-12,16,24,40,56}"
export V14_D_MODEL="${V14_D_MODEL:-96}"
export V14_ATTENTION_HEADS="${V14_ATTENTION_HEADS:-4}"
export V14_DECODER_LAYERS="${V14_DECODER_LAYERS:-2}"
export V14_FEEDFORWARD_DIM="${V14_FEEDFORWARD_DIM:-192}"
export V14_DROPOUT="${V14_DROPOUT:-0.15}"
export V14_DELTA_GAIN="${V14_DELTA_GAIN:-1.0}"
export V14_TIME_MASK_PROBABILITY="${V14_TIME_MASK_PROBABILITY:-0.15}"
export V14_MODALITY_DROP_PROBABILITY="${V14_MODALITY_DROP_PROBABILITY:-0.25}"
export V14_TIMING_SIGMA_HOURS="${V14_TIMING_SIGMA_HOURS:-18}"
export V14_PHYSICAL_PRIOR_MIN="${V14_PHYSICAL_PRIOR_MIN:-1.0}"
export V14_PHYSICAL_PRIOR_MAX="${V14_PHYSICAL_PRIOR_MAX:-4.0}"
export V14_PHYSICAL_PRIOR_INIT="${V14_PHYSICAL_PRIOR_INIT:-2.0}"
export V14_MAXIMUM_BLEND="${V14_MAXIMUM_BLEND:-0.50}"
export V14_INITIAL_BLEND="${V14_INITIAL_BLEND:-0.05}"
export V14_CORRECTION_CAP_MULTIPLIER="${V14_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V14_HINDCAST_WEIGHT_START="${V14_HINDCAST_WEIGHT_START:-0.50}"
export V14_HINDCAST_WEIGHT_END="${V14_HINDCAST_WEIGHT_END:-0.10}"
export V14_HINDCAST_DECAY_EPOCHS="${V14_HINDCAST_DECAY_EPOCHS:-8}"
export V14_ALIGNMENT_SIGMA_DEG="${V14_ALIGNMENT_SIGMA_DEG:-12}"
export V14_CORRECTION_L2_WEIGHT="${V14_CORRECTION_L2_WEIGHT:-0.10}"
export V14_GATE_L1_WEIGHT="${V14_GATE_L1_WEIGHT:-0.01}"
export V14_SPEED_SMOOTHNESS_WEIGHT="${V14_SPEED_SMOOTHNESS_WEIGHT:-0.02}"
export V14_DEFORMABLE_POINTS="${V14_DEFORMABLE_POINTS:-8}"
export V14_MAXIMUM_TIME_OFFSET_HOURS="${V14_MAXIMUM_TIME_OFFSET_HOURS:-12}"
export V14_MAXIMUM_LONGITUDE_OFFSET_CELLS="${V14_MAXIMUM_LONGITUDE_OFFSET_CELLS:-1.5}"
export V14_DENSE_KERNEL_TIME_FRAMES="${V14_DENSE_KERNEL_TIME_FRAMES:-0.75}"
export V14_DENSE_KERNEL_LONGITUDE_CELLS="${V14_DENSE_KERNEL_LONGITUDE_CELLS:-0.75}"
export V14_GRADIENT_CLIP="${V14_GRADIENT_CLIP:-1.0}"
export V14_WEIGHT_DECAY="${V14_WEIGHT_DECAY:-0.03}"
export V14_WARMUP_EPOCHS="${V14_WARMUP_EPOCHS:-3}"
export V14_MIN_LR="${V14_MIN_LR:-1e-6}"
export V14_EARLY_STOP_PATIENCE="${V14_EARLY_STOP_PATIENCE:-6}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "smoke" ]]; then
  exec python src_taeukjung/smoke_solar_deformable_timing_v14.py
fi

run_experiment() {
  local name="$1"
  local alignment_weight="$2"
  local output_dir="${V14_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_deformable_timing_v14.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_deformable_timing_v14.py"
      ;;
    *)
      echo "usage: $0 [smoke|train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: image=${IMAGE_SIZE} grid=${V14_GRID_ROWS}x${V14_GRID_COLUMNS} points=${V14_DEFORMABLE_POINTS} backmapping=${alignment_weight}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V14_ALIGNMENT_WEIGHT="${alignment_weight}" \
    python "${program}"
}

EXPERIMENTS="${V14_EXPERIMENTS:-v14_deformable_full v14_deformable_no_backmapping}"

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/smoke_solar_deformable_timing_v14.py
fi

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    v14_deformable_full)
      run_experiment "${experiment}" 0.01
      ;;
    v14_deformable_no_backmapping)
      run_experiment "${experiment}" 0.0
      ;;
    *)
      echo "unknown V14 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_deformable_timing_v14.py
fi
