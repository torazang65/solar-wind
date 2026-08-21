#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V15_ABLATION_ROOT="${V15_ABLATION_ROOT:-/home/jovyan/outputs/solar_peak_event_v15_seed${SEED}}"
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

export V15_GRID_ROWS="${V15_GRID_ROWS:-2}"
export V15_GRID_COLUMNS="${V15_GRID_COLUMNS:-8}"
export V15_UNET_CHANNELS="${V15_UNET_CHANNELS:-12,16,24,40,56}"
export V15_D_MODEL="${V15_D_MODEL:-96}"
export V15_ATTENTION_HEADS="${V15_ATTENTION_HEADS:-4}"
export V15_DECODER_LAYERS="${V15_DECODER_LAYERS:-2}"
export V15_FEEDFORWARD_DIM="${V15_FEEDFORWARD_DIM:-192}"
export V15_DROPOUT="${V15_DROPOUT:-0.15}"
export V15_DELTA_GAIN="${V15_DELTA_GAIN:-1.0}"
export V15_TIME_MASK_PROBABILITY="${V15_TIME_MASK_PROBABILITY:-0.15}"
export V15_MODALITY_DROP_PROBABILITY="${V15_MODALITY_DROP_PROBABILITY:-0.25}"
export V15_TIMING_SIGMA_HOURS="${V15_TIMING_SIGMA_HOURS:-18}"
export V15_PHYSICAL_PRIOR_MIN="${V15_PHYSICAL_PRIOR_MIN:-1.0}"
export V15_PHYSICAL_PRIOR_MAX="${V15_PHYSICAL_PRIOR_MAX:-4.0}"
export V15_PHYSICAL_PRIOR_INIT="${V15_PHYSICAL_PRIOR_INIT:-2.0}"
export V15_MAXIMUM_BLEND="${V15_MAXIMUM_BLEND:-0.50}"
export V15_INITIAL_BLEND="${V15_INITIAL_BLEND:-0.05}"
export V15_CORRECTION_CAP_MULTIPLIER="${V15_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V15_HINDCAST_WEIGHT_START="${V15_HINDCAST_WEIGHT_START:-0.50}"
export V15_HINDCAST_WEIGHT_END="${V15_HINDCAST_WEIGHT_END:-0.10}"
export V15_HINDCAST_DECAY_EPOCHS="${V15_HINDCAST_DECAY_EPOCHS:-8}"
export V15_ALIGNMENT_WEIGHT="${V15_ALIGNMENT_WEIGHT:-0.0}"
export V15_ALIGNMENT_SIGMA_DEG="${V15_ALIGNMENT_SIGMA_DEG:-12}"
export V15_CORRECTION_L2_WEIGHT="${V15_CORRECTION_L2_WEIGHT:-0.10}"
export V15_GATE_L1_WEIGHT="${V15_GATE_L1_WEIGHT:-0.01}"
export V15_SPEED_SMOOTHNESS_WEIGHT="${V15_SPEED_SMOOTHNESS_WEIGHT:-0.02}"
export V15_DEFORMABLE_POINTS="${V15_DEFORMABLE_POINTS:-8}"
export V15_MAXIMUM_TIME_OFFSET_HOURS="${V15_MAXIMUM_TIME_OFFSET_HOURS:-12}"
export V15_MAXIMUM_LONGITUDE_OFFSET_CELLS="${V15_MAXIMUM_LONGITUDE_OFFSET_CELLS:-1.5}"
export V15_DENSE_KERNEL_TIME_FRAMES="${V15_DENSE_KERNEL_TIME_FRAMES:-0.75}"
export V15_DENSE_KERNEL_LONGITUDE_CELLS="${V15_DENSE_KERNEL_LONGITUDE_CELLS:-0.75}"
export V15_PEAK_HIDDEN_DIM="${V15_PEAK_HIDDEN_DIM:-96}"
export V15_PEAK_CURVE_SIGMA_STEPS="${V15_PEAK_CURVE_SIGMA_STEPS:-1.25}"
export V15_PEAK_VALUE_MIN="${V15_PEAK_VALUE_MIN:-0.25}"
export V15_PEAK_VALUE_MAX="${V15_PEAK_VALUE_MAX:-0.90}"
export V15_MAXIMUM_PEAK_BLEND="${V15_MAXIMUM_PEAK_BLEND:-0.30}"
export V15_INITIAL_PEAK_BLEND="${V15_INITIAL_PEAK_BLEND:-0.05}"
export V15_PEAK_CORRECTION_CAP_MULTIPLIER="${V15_PEAK_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V15_PEAK_LABEL_SIGMA_STEPS="${V15_PEAK_LABEL_SIGMA_STEPS:-1.0}"
export V15_PEAK_TIMING_PROMINENCE_KMS="${V15_PEAK_TIMING_PROMINENCE_KMS:-60}"
export V15_PEAK_TIMING_MINIMUM_WEIGHT="${V15_PEAK_TIMING_MINIMUM_WEIGHT:-0.10}"
export V15_GRADIENT_CLIP="${V15_GRADIENT_CLIP:-1.0}"
export V15_WEIGHT_DECAY="${V15_WEIGHT_DECAY:-0.03}"
export V15_WARMUP_EPOCHS="${V15_WARMUP_EPOCHS:-3}"
export V15_MIN_LR="${V15_MIN_LR:-1e-6}"
export V15_EARLY_STOP_PATIENCE="${V15_EARLY_STOP_PATIENCE:-6}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "smoke" ]]; then
  exec python src_taeukjung/smoke_solar_peak_event_v15.py
fi

run_experiment() {
  local name="$1"
  local time_weight="$2"
  local value_weight="$3"
  local output_dir="${V15_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_peak_event_v15.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_peak_event_v15.py"
      ;;
    *)
      echo "usage: $0 [smoke|train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: peak_time_weight=${time_weight} peak_value_weight=${value_weight} max_peak_blend=${V15_MAXIMUM_PEAK_BLEND}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V15_PEAK_TIME_LOSS_WEIGHT="${time_weight}" \
    V15_PEAK_VALUE_LOSS_WEIGHT="${value_weight}" \
    python "${program}"
}

EXPERIMENTS="${V15_EXPERIMENTS:-v15_peak_joint v15_peak_time_strong}"

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/smoke_solar_peak_event_v15.py
fi

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    v15_peak_joint)
      run_experiment "${experiment}" 0.05 0.25
      ;;
    v15_peak_time_strong)
      run_experiment "${experiment}" 0.10 0.25
      ;;
    *)
      echo "unknown V15 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_peak_event_v15.py
fi
