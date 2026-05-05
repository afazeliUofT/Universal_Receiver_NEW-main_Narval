#!/usr/bin/env bash
#SBATCH --job-name=upair-pub-mu16
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --array=0-35%6
#SBATCH --signal=B:TERM@600

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/rsadve1/scratch/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-/home/rsadve1/scratch/.venvGRAND}"
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
export PYTHONUNBUFFERED=1
export PIP_NO_USER=1
export TF_GPU_ALLOCATOR="${TF_GPU_ALLOCATOR:-cuda_malloc_async}"
unset PYTHONPATH
export TF_CPP_MIN_LOG_LEVEL=1
export TF_FORCE_GPU_ALLOW_GROWTH=true
export MPLBACKEND=Agg

# shellcheck disable=SC1091
source "${VENV_PATH}/bin/activate"

variants=(
  main_d96_b4_r2
  shallow_d96_b2_r2
  deep_d96_b6_r2
  narrow_d64_b4_r2
  wide_d128_b4_r2
  mlpwide_d96_b4_r4
)
dmrs_cases=(1dmrs 2dmrs)
IFS=',' read -r -a seeds <<< "${UPAIR_PUBLICATION_SEEDS:-7,17,29}"

idx="${SLURM_ARRAY_TASK_ID}"
variant_count="${#variants[@]}"
case_count="${#dmrs_cases[@]}"
seed_count="${#seeds[@]}"
tasks_per_seed=$((case_count * variant_count))
total_tasks=$((seed_count * tasks_per_seed))

if (( idx >= total_tasks )); then
  echo "[SLURM] task=${idx} is outside configured seed/case/variant grid of ${total_tasks}; exiting."
  exit 0
fi

seed_idx=$((idx / tasks_per_seed))
within_seed_idx=$((idx % tasks_per_seed))
case_idx=$((within_seed_idx / variant_count))
variant_idx=$((within_seed_idx % variant_count))

dmrs_case="${dmrs_cases[$case_idx]}"
variant="${variants[$variant_idx]}"
seed="${seeds[$seed_idx]}"

echo "[SLURM] task=${idx}/${total_tasks} seed=${seed} dmrs_case=${dmrs_case} variant=${variant}"
nvidia-smi

extra_args=()
if [[ "${UPAIR_COMPREHENSIVE_EVAL_ONLY:-0}" == "1" ]]; then
  extra_args+=(--eval-only)
fi
if [[ "${UPAIR_COMPREHENSIVE_FORCE:-0}" == "1" ]]; then
  extra_args+=(--force)
fi
if [[ -n "${UPAIR_COMPREHENSIVE_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra_args+=(${UPAIR_COMPREHENSIVE_EXTRA_ARGS})
fi

child_pid=""
forward_term() {
  echo "[SLURM] received pre-timeout signal; forwarding TERM to Python worker ${child_pid}" >&2
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" || true
  fi
  exit 143
}
trap forward_term TERM INT

python scripts/run_comprehensive_mu32_ablation.py \
  --config configs/twc_comprehensive_mu32_base.yaml \
  --seeds "${seed}" \
  --dmrs-cases "${dmrs_case}" \
  --variants "${variant}" \
  --no-global-summary \
  "${extra_args[@]}" &
child_pid=$!
wait "${child_pid}"
