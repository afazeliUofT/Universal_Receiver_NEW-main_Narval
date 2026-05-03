#!/usr/bin/env bash
#SBATCH --job-name=upair-batchprobe
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvUPAIR}"
STDENV_MODULE="${STDENV_MODULE:-StdEnv/2023}"
GCC_MODULE="${GCC_MODULE:-gcc/12.3}"
PYTHON_MODULE="${PYTHON_MODULE:-python/3.11.5}"
CUDA_MODULE="${CUDA_MODULE:-cuda/12.2}"
cd "${PROJECT_ROOT}"
mkdir -p logs

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

module --force purge || true
module load "${STDENV_MODULE}"
module load "${GCC_MODULE}"
module load "${PYTHON_MODULE}"
if [[ -n "${CUDA_MODULE}" ]]; then
  module load "${CUDA_MODULE}"
fi

export PYTHONNOUSERSITE=1
export PIP_NO_USER=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
unset PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

nvidia-smi
python scripts/find_max_safe_batch_narval.py \
  --batch-sizes "${UPAIR_BATCH_PROBE_SIZES:-64,32,16,8,4}" \
  --phase "${UPAIR_BATCH_PROBE_PHASE:-both}" \
  --search-order "${UPAIR_BATCH_PROBE_SEARCH_ORDER:-descending}" \
  --num-rx-ant "${UPAIR_BATCH_PROBE_NUM_RX_ANT:-16}" \
  --steps "${UPAIR_BATCH_PROBE_STEPS:-350}" \
  --eval-every "${UPAIR_BATCH_PROBE_EVAL_EVERY:-175}" \
  --val-steps "${UPAIR_BATCH_PROBE_VAL_STEPS:-16}" \
  --eval-num-users "${UPAIR_BATCH_PROBE_EVAL_USERS:-4}" \
  --eval-ebno-points "${UPAIR_BATCH_PROBE_EBNO:-0,8}" \
  --eval-num-batches "${UPAIR_BATCH_PROBE_EVAL_BATCHES:-2}" \
  --eval-cov-batches "${UPAIR_BATCH_PROBE_COV_BATCHES:-2}" \
  --per-case-timeout-min "${UPAIR_BATCH_PROBE_TIMEOUT_MIN:-45}" \
  ${UPAIR_BATCH_PROBE_EXTRA_ARGS:-}
