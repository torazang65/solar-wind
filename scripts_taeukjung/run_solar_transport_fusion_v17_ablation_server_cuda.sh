#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ACTION="${1:-train}"

cd "${REPO_ROOT}"

export SEED="${SEED:-777}"
export DATA_ROOT="${DATA_ROOT:-/home/jovyan/public_dataset/competition_dataset_6h}"
export CACHE_DIR="${CACHE_DIR:-/home/jovyan/outputs/cache_taeukjung}"
export V17_ABLATION_ROOT="${V17_ABLATION_ROOT:-/home/jovyan/outputs/solar_transport_fusion_v17_seed${SEED}}"
export PYTHONPATH="${REPO_ROOT}/src_taeukjung:${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

export IMAGE_SIZE="${IMAGE_SIZE:-64}"
export IMAGE_NORM="${IMAGE_NORM:-linear}"
export SOFT_CUBIC_STRENGTH="${SOFT_CUBIC_STRENGTH:-0.0}"
export SOLAR_DISK_MASK="${SOLAR_DISK_MASK:-1}"
export SOLAR_DISK_RADIUS_FRACTION="${SOLAR_DISK_RADIUS_FRACTION:-0.49}"
export BATCH_SIZE="${BATCH_SIZE:-128}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export CHAIN_BALANCED_SAMPLING="${CHAIN_BALANCED_SAMPLING:-0}"

export V17_COLUMN_DIM="${V17_COLUMN_DIM:-32}"
export V17_LONGITUDE_KERNEL_SIZE="${V17_LONGITUDE_KERNEL_SIZE:-5}"
export V17_SPEED_EXPERTS_KMS="${V17_SPEED_EXPERTS_KMS:-300,400,500,650,800}"
export V17_TRANSPORT_SIGMA_HOURS="${V17_TRANSPORT_SIGMA_HOURS:-15}"
export V17_EFFECTIVE_DISTANCE_HOURS_AT_1000_KMS="${V17_EFFECTIVE_DISTANCE_HOURS_AT_1000_KMS:-41.6}"
export V17_MINIMUM_DELAY_HOURS="${V17_MINIMUM_DELAY_HOURS:-48}"
export V17_MAXIMUM_DELAY_HOURS="${V17_MAXIMUM_DELAY_HOURS:-144}"
export V17_TRANSPORT_STRENGTH="${V17_TRANSPORT_STRENGTH:-0.50}"
export V17_CORRECTION_CAP_MULTIPLIER="${V17_CORRECTION_CAP_MULTIPLIER:-1.0}"
export V17_TIME_MASK_PROBABILITY="${V17_TIME_MASK_PROBABILITY:-0.10}"
export V17_MODALITY_DROP_PROBABILITY="${V17_MODALITY_DROP_PROBABILITY:-0.10}"
export V17_TRANSPORT_LR="${V17_TRANSPORT_LR:-3e-4}"
export V17_FUSION_LR="${V17_FUSION_LR:-3e-4}"
export V17_JOINT_LR="${V17_JOINT_LR:-5e-5}"
export V17_HINDCAST_WEIGHT="${V17_HINDCAST_WEIGHT:-0.50}"
export V17_FUTURE_TRANSPORT_WEIGHT="${V17_FUTURE_TRANSPORT_WEIGHT:-0.10}"
export V17_CORRECTION_L2_WEIGHT="${V17_CORRECTION_L2_WEIGHT:-0.01}"
export V17_SMOOTHNESS_WEIGHT="${V17_SMOOTHNESS_WEIGHT:-0.01}"
export V17_ENTROPY_WEIGHT="${V17_ENTROPY_WEIGHT:-0.001}"
export V17_WEIGHT_DECAY="${V17_WEIGHT_DECAY:-0.02}"
export V17_GRADIENT_CLIP="${V17_GRADIENT_CLIP:-1.0}"
export V17_EARLY_STOP_PATIENCE="${V17_EARLY_STOP_PATIENCE:-5}"

python -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable"; print(torch.cuda.get_device_name(0))'

if [[ "${ACTION}" == "smoke" ]]; then
  exec python src_taeukjung/smoke_solar_transport_fusion_v17.py
fi

run_experiment() {
  local name="$1"
  local scramble_images="$2"
  local transport_epochs="$3"
  local fusion_epochs="$4"
  local joint_epochs="$5"
  local output_dir="${V17_ABLATION_ROOT}/${name}"
  local program

  case "${ACTION}" in
    train)
      program="src_taeukjung/train_solar_transport_fusion_v17.py"
      ;;
    infer)
      program="src_taeukjung/inference_solar_transport_fusion_v17.py"
      ;;
    *)
      echo "usage: $0 [smoke|train|infer]" >&2
      exit 2
      ;;
  esac

  echo
  echo "[${ACTION}] ${name}: scramble=${scramble_images} phases=${transport_epochs}/${fusion_epochs}/${joint_epochs}"
  env \
    OUTPUT_DIR="${output_dir}" \
    V17_SCRAMBLE_IMAGES="${scramble_images}" \
    V17_TRANSPORT_EPOCHS="${transport_epochs}" \
    V17_FUSION_EPOCHS="${fusion_epochs}" \
    V17_JOINT_EPOCHS="${joint_epochs}" \
    python "${program}"
}

EXPERIMENTS="${V17_EXPERIMENTS:-v17_native v17_scrambled v17_no_pretrain}"

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/smoke_solar_transport_fusion_v17.py
fi

for experiment in ${EXPERIMENTS}; do
  case "${experiment}" in
    v17_native)
      run_experiment "${experiment}" 0 6 10 4
      ;;
    v17_scrambled)
      run_experiment "${experiment}" 1 6 10 4
      ;;
    v17_no_pretrain)
      run_experiment "${experiment}" 0 0 12 4
      ;;
    *)
      echo "unknown V17 experiment: ${experiment}" >&2
      exit 2
      ;;
  esac
done

if [[ "${ACTION}" == "train" ]]; then
  python src_taeukjung/summarize_solar_transport_fusion_v17.py
fi
