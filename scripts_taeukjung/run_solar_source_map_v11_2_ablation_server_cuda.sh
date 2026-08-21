#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V112_ABLATION_ROOT="${V112_ABLATION_ROOT:-/home/jovyan/outputs/solar_source_map_v11_2_ablation_seed${SEED}}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export EPOCHS="${EPOCHS:-35}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export LEARNING_RATE="${LEARNING_RATE:-3e-5}"

export V112_D_MODEL="${V112_D_MODEL:-128}"
export V112_DROPOUT="${V112_DROPOUT:-0.10}"
export V112_DELTA_GAIN="${V112_DELTA_GAIN:-1.0}"
export V112_TIME_MASK_PROBABILITY="${V112_TIME_MASK_PROBABILITY:-0.15}"
export V112_MODALITY_DROP_PROBABILITY="${V112_MODALITY_DROP_PROBABILITY:-0.25}"
export V112_KERNEL_SIGMA_HOURS="${V112_KERNEL_SIGMA_HOURS:-12}"
export V112_TRANSIT_RESIDUAL_HOURS="${V112_TRANSIT_RESIDUAL_HOURS:-24}"
export V112_HINDCAST_WEIGHT_START="${V112_HINDCAST_WEIGHT_START:-0.70}"
export V112_HINDCAST_WEIGHT_END="${V112_HINDCAST_WEIGHT_END:-0.10}"
export V112_HINDCAST_DECAY_EPOCHS="${V112_HINDCAST_DECAY_EPOCHS:-8}"
export V112_ALIGNMENT_WEIGHT="${V112_ALIGNMENT_WEIGHT:-0.02}"
export V112_ALIGNMENT_SIGMA_DEG="${V112_ALIGNMENT_SIGMA_DEG:-20}"
export V112_TRANSIT_RESIDUAL_L2="${V112_TRANSIT_RESIDUAL_L2:-0.003}"
export V112_SURGE_WEIGHT="${V112_SURGE_WEIGHT:-0.02}"
export V112_PHYSICAL_LR_MULT="${V112_PHYSICAL_LR_MULT:-100}"
export V112_WARMUP_EPOCHS="${V112_WARMUP_EPOCHS:-3}"
export V112_MIN_LR="${V112_MIN_LR:-1e-6}"
export V112_WEIGHT_DECAY="${V112_WEIGHT_DECAY:-0.01}"
export V112_EARLY_STOP_PATIENCE="${V112_EARLY_STOP_PATIENCE:-15}"
export V112_CHAIN_BALANCED="${V112_CHAIN_BALANCED:-0}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

run_experiment() {
  local name="$1"
  local image_size="$2"
  local grid_columns="$3"
  local batch_size="$4"
  local consistency_weight="$5"
  local output_dir="${V112_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_source_map_v11_2.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_source_map_v11_2.py"
      ;;
    diagnose)
      program="src_taeukjung/diagnose_solar_source_map_v11_2.py"
      ;;
    *)
      echo "usage: $0 [train|infer|diagnose]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: image=${image_size} grid=2x${grid_columns} batch=${batch_size} consistency=${consistency_weight}"
  env \
    OUTPUT_DIR="${output_dir}" \
    IMAGE_SIZE="${image_size}" \
    BATCH_SIZE="${batch_size}" \
    V112_GRID_ROWS=2 \
    V112_GRID_COLUMNS="${grid_columns}" \
    V112_CONSISTENCY_WEIGHT="${consistency_weight}" \
    python "${program}"
}

EXPERIMENTS="${V112_EXPERIMENTS:-exp1_64_2x4_maskfix exp3_128_2x8_maskfix exp4_128_2x8_consistency}"

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    exp1_64_2x4_maskfix)
      run_experiment "${experiment}" 64 4 256 0
      ;;
    exp3_128_2x8_maskfix)
      run_experiment "${experiment}" 128 8 64 0
      ;;
    exp4_128_2x8_consistency)
      run_experiment "${experiment}" 128 8 32 0.05
      ;;
    *)
      echo "unknown V11.2 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_source_map_v11_2.py
fi
