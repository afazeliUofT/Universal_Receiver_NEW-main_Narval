#!/usr/bin/env bash
#SBATCH --job-name=upair-probe-narval
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1

set -euo pipefail

SCRATCH_ROOT="${SCRATCH_ROOT:-/home/rsadve1/scratch}"
PROJECT_ROOT="${PROJECT_ROOT:-${SCRATCH_ROOT}/Universal_Receiver_NEW-main_Narval}"
VENV_PATH="${VENV_PATH:-${SCRATCH_ROOT}/.venvUPAIR}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}}"

cd "${PROJECT_ROOT}"
mkdir -p logs "${OUTPUT_DIR}"

if [[ -f /etc/profile.d/modules.sh ]]; then
  # shellcheck disable=SC1091
  source /etc/profile.d/modules.sh
fi

echo "=== Narval GPU-node probe: current/default module state ==="
nvidia-smi || true

python3 "${PROJECT_ROOT}/probe_narval_environment.py" \
  --label "gpu-${SLURM_JOB_ID:-manual}" \
  --scratch-root "${SCRATCH_ROOT}" \
  --project-root "${PROJECT_ROOT}" \
  --venv-path "${VENV_PATH}" \
  --output "${OUTPUT_DIR}/narval_env_probe_gpu_${SLURM_JOB_ID:-manual}.json"

echo "Send back ${OUTPUT_DIR}/narval_env_probe_gpu_${SLURM_JOB_ID:-manual}.json"
